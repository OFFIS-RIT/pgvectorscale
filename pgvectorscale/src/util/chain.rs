//! This module defines the `ChainTape` data structure, which is used to store large data items that
//! are too big to fit in a single page.  See `Tape` for a simpler version that assumes each data
//! item fits in a single page.
//!
//! All page entries begin with a header that contains an item pointer to the next chunk in the chain,
//! if applicable.  The last chunk in the chain has an invalid item pointer.
//!
//! The implementation supports an append-only sequence of writes via `ChainTapeWriter` and reads
//! via `ChainTapeReader`.  The writer returns an `ItemPointer` that can be used to read the data
//! back.  Reads are done via an iterator that returns `ReadableBuffer` objects for the segments
//! of the data.

use pgrx::{
    pg_sys::{BlockNumber, InvalidBlockNumber},
    PgRelation,
};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::stats::{StatsNodeRead, StatsNodeWrite};

use super::{
    page::{PageType, ReadablePage, WritablePage},
    ItemPointer, ReadableBuffer,
};

#[derive(Clone, PartialEq, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
struct ChainItemHeader {
    next: ItemPointer,
}

const CHAIN_ITEM_HEADER_SIZE: usize = std::mem::size_of::<ArchivedChainItemHeader>();

pub struct ChainTapeWriter<'a, S: StatsNodeWrite> {
    page_type: PageType,
    index: &'a PgRelation,
    current: BlockNumber,
    stats: &'a mut S,
}

impl<'a, S: StatsNodeWrite> ChainTapeWriter<'a, S> {
    /// Create a ChainTape that starts writing on a new page.
    pub fn new(index: &'a PgRelation, page_type: PageType, stats: &'a mut S) -> Self {
        assert!(page_type.is_chained());
        let page = WritablePage::new(index, page_type);
        let block_number = page.get_block_number();
        page.commit();
        Self {
            page_type,
            index,
            current: block_number,
            stats,
        }
    }

    /// Stage a replacement root, keeping it exclusively locked until the caller commits it.
    /// Items start at successive root offsets; all but the last must fit on the root.
    /// Any overflow is fully written before returning, but remains unreachable until commit.
    /// Dropping the returned page instead leaves an existing root unchanged.
    /// For first-time writes, the relation must allocate the requested root block next.
    pub fn prepare_root(
        index: &'a PgRelation,
        page_type: PageType,
        stats: &mut S,
        block_number: BlockNumber,
        first_time: bool,
        items: &[&[u8]],
    ) -> WritablePage<'a> {
        assert!(page_type.is_chained());
        assert!(!items.is_empty());
        let mut root = if first_time {
            WritablePage::new(index, page_type)
        } else {
            let mut page = WritablePage::modify(index, block_number);
            // Reinitialize only the Generic WAL copy, never publish an empty root.
            page.reinit(page_type);
            page
        };
        assert_eq!(root.get_block_number(), block_number);

        for (i, data) in items.iter().enumerate() {
            assert!(!data.is_empty());
            let free_space = root.get_aligned_free_space();
            assert!(free_space > CHAIN_ITEM_HEADER_SIZE);
            let data_size = data.len().min(free_space - CHAIN_ITEM_HEADER_SIZE);
            let next = if data_size < data.len() {
                assert_eq!(i, items.len() - 1, "only the last root item may overflow");
                let mut suffix = ChainTapeWriter::new(index, page_type, stats);
                suffix.write(&data[data_size..])
            } else {
                ItemPointer::new_invalid()
            };
            let header_bytes = rkyv::to_bytes::<_, 256>(&ChainItemHeader { next }).unwrap();
            let combined = [header_bytes.as_slice(), &data[..data_size]].concat();
            let offset = root.add_item(&combined);
            assert_eq!(offset as usize, i + 1);
        }

        // Old suffixes (and suffixes from aborted replacements) are not reclaimed.
        root
    }

    /// Write chained data to the tape, returning an `ItemPointer` to the start of the data.
    pub fn write(&mut self, mut data: &[u8]) -> super::ItemPointer {
        let mut current_page = WritablePage::modify(self.index, self.current);

        // If there isn't enough space for the header plus some data, start a new page.
        if current_page.get_aligned_free_space() < CHAIN_ITEM_HEADER_SIZE + 1 {
            current_page = WritablePage::new(self.index, self.page_type);
            self.current = current_page.get_block_number();
        }

        // ItemPointer to the first item in the chain.
        let mut result: Option<super::ItemPointer> = None;

        // Write the data in chunks, creating new pages as needed.
        while CHAIN_ITEM_HEADER_SIZE + data.len() > current_page.get_aligned_free_space() {
            let next_page = WritablePage::new(self.index, self.page_type);
            let header = ChainItemHeader {
                next: ItemPointer::new(next_page.get_block_number(), 1),
            };
            let header_bytes = rkyv::to_bytes::<_, 256>(&header).unwrap();
            let data_size = current_page.get_aligned_free_space() - CHAIN_ITEM_HEADER_SIZE;
            let chunk = &data[..data_size];
            let combined = [header_bytes.as_slice(), chunk].concat();
            let offset_number = current_page.add_item(combined.as_ref());
            result.get_or_insert_with(|| {
                ItemPointer::new(current_page.get_block_number(), offset_number)
            });
            current_page.commit();
            self.stats.record_write();
            current_page = next_page;
            data = &data[data_size..];
        }

        // Write the last chunk of data.
        let header = ChainItemHeader {
            next: ItemPointer::new_invalid(),
        };
        let header_bytes = rkyv::to_bytes::<_, 256>(&header).unwrap();
        let combined = [header_bytes.as_slice(), data].concat();
        let offset_number = current_page.add_item(combined.as_ref());
        let result = result
            .unwrap_or_else(|| ItemPointer::new(current_page.get_block_number(), offset_number));
        self.current = current_page.get_block_number();
        current_page.commit();
        self.stats.record_write();

        result
    }
}

pub struct ChainItemReader<'a, S: StatsNodeRead> {
    page_type: PageType,
    index: &'a PgRelation,
    stats: &'a mut S,
}

impl<'a, S: StatsNodeRead> ChainItemReader<'a, S> {
    pub fn new(index: &'a PgRelation, page_type: PageType, stats: &'a mut S) -> Self {
        assert!(page_type.is_chained());
        Self {
            page_type,
            index,
            stats,
        }
    }

    pub fn read(&'a mut self, ip: ItemPointer) -> ChainItemIterator<'a, S> {
        ChainItemIterator {
            index: self.index,
            ip,
            page_type: self.page_type,
            stats: self.stats,
        }
    }
}

pub struct ChainItemIterator<'a, S: StatsNodeRead> {
    index: &'a PgRelation,
    ip: ItemPointer,
    page_type: PageType,
    stats: &'a mut S,
}

impl<'a, S: StatsNodeRead> Iterator for ChainItemIterator<'a, S> {
    type Item = ReadableBuffer<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ip.block_number == InvalidBlockNumber {
            return None;
        }

        unsafe {
            let page = ReadablePage::read(self.index, self.ip.block_number);
            self.stats.record_read();
            assert!(page.get_type() == self.page_type);
            let mut item = page.get_item_unchecked(self.ip.offset);
            let slice = item.get_data_slice();
            assert!(slice.len() > CHAIN_ITEM_HEADER_SIZE);
            let header_slice = &slice[..CHAIN_ITEM_HEADER_SIZE];

            let header = rkyv::check_archived_root::<ChainItemHeader>(header_slice).unwrap();
            self.ip = ItemPointer::new(header.next.block_number, header.next.offset);

            item.advance(CHAIN_ITEM_HEADER_SIZE);

            Some(item)
        }
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::{
        pg_sys::{self, BLCKSZ},
        pg_test, Spi,
    };

    use crate::access_method::stats::InsertStats;

    use super::*;

    fn make_test_relation() -> PgRelation {
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING diskann(encoding)
                WITH (num_neighbors=30);",
        )
        .unwrap();

        let index_oid = Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")
            .unwrap()
            .expect("oid was null");
        unsafe { PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid)) }
    }

    fn read_root_items(index: &PgRelation, expected: &[&[u8]]) {
        let mut stats = InsertStats::default();
        // Match metadata fetch: retain a root share lock while reading both chains.
        let root = unsafe { ReadablePage::read(index, 0) };
        assert_eq!(root.get_type(), PageType::Meta);
        for (i, data) in expected.iter().enumerate() {
            let mut reader = ChainItemReader::new(index, PageType::Meta, &mut stats);
            let actual: Vec<u8> = reader
                .read(ItemPointer::new(0, (i + 1) as pg_sys::OffsetNumber))
                .flat_map(|item| item.get_data_slice().to_vec())
                .collect();
            assert_eq!(actual.as_slice(), *data);
        }
    }

    #[pg_test]
    fn test_chain_root_replacement() {
        let index = make_test_relation();
        let mut stats = InsertStats::default();
        let header = b"root header";
        for size in [3 * BLCKSZ as usize, 1, BLCKSZ as usize, 17] {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            ChainTapeWriter::prepare_root(
                &index,
                PageType::Meta,
                &mut stats,
                0,
                false,
                &[header, &data],
            )
            .commit();
            read_root_items(&index, &[header, &data]);
        }
    }

    #[pg_test]
    fn test_chain_root_abort_before_publication() {
        let index = make_test_relation();
        let mut stats = InsertStats::default();
        let old_data = vec![42; 3 * BLCKSZ as usize];
        ChainTapeWriter::prepare_root(
            &index,
            PageType::Meta,
            &mut stats,
            0,
            false,
            &[b"old header", &old_data],
        )
        .commit();

        let new_data = vec![99; 4 * BLCKSZ as usize];
        let root = ChainTapeWriter::prepare_root(
            &index,
            PageType::Meta,
            &mut stats,
            0,
            false,
            &[b"new header", &new_data],
        );
        // Suffix writes have finished, but abort the root's Generic WAL operation.
        drop(root);
        read_root_items(&index, &[b"old header", &old_data]);

        ChainTapeWriter::prepare_root(
            &index,
            PageType::Meta,
            &mut stats,
            0,
            false,
            &[b"new header", &new_data],
        )
        .commit();
        read_root_items(&index, &[b"new header", &new_data]);
    }

    #[pg_test]
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::manual_slice_fill)]
    fn test_chain_tape() {
        let mut rstats = InsertStats::default();
        let mut wstats = InsertStats::default();

        let index = make_test_relation();
        {
            // ChainTape can be used for small items too
            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for i in 0..100 {
                let data = format!("hello world {i}");
                let ip = tape.write(data.as_bytes());
                let mut reader = ChainItemReader::new(&index, PageType::SbqMeans, &mut rstats);

                let mut iter = reader.read(ip);
                let item = iter.next().unwrap();
                assert_eq!(item.get_data_slice(), data.as_bytes());
                assert!(iter.next().is_none());
            }
        }

        for data_size in BLCKSZ - 100..BLCKSZ + 100 {
            // Exhaustively test around the neighborhood of a page size
            let mut bigdata = vec![0u8; data_size as usize];
            for i in 0..bigdata.len() {
                bigdata[i] = (i % 256) as u8;
            }

            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for _ in 0..10 {
                let ip = tape.write(&bigdata);
                let mut count = 0;
                let mut reader = ChainItemReader::new(&index, PageType::SbqMeans, &mut rstats);
                for item in reader.read(ip) {
                    assert_eq!(item.get_data_slice(), &bigdata[count..count + item.len]);
                    count += item.len;
                }
                assert_eq!(count, bigdata.len());
            }
        }

        for data_size in (2 * BLCKSZ - 100)..(2 * BLCKSZ + 100) {
            // Exhaustively test around the neighborhood of a 2-page size
            let mut bigdata = vec![0u8; data_size as usize];
            for i in 0..bigdata.len() {
                bigdata[i] = (i % 256) as u8;
            }

            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for _ in 0..10 {
                let ip = tape.write(&bigdata);
                let mut count = 0;
                let mut reader = ChainItemReader::new(&index, PageType::SbqMeans, &mut rstats);
                for item in reader.read(ip) {
                    assert_eq!(item.get_data_slice(), &bigdata[count..count + item.len]);
                    count += item.len;
                }
                assert_eq!(count, bigdata.len());
            }
        }

        for data_size in (3 * BLCKSZ - 100)..(3 * BLCKSZ + 100) {
            // Exhaustively test around the neighborhood of a 3-page size
            let mut bigdata = vec![0u8; data_size as usize];
            for i in 0..bigdata.len() {
                bigdata[i] = (i % 256) as u8;
            }

            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for _ in 0..10 {
                let ip = tape.write(&bigdata);
                let mut count = 0;
                let mut reader = ChainItemReader::new(&index, PageType::SbqMeans, &mut rstats);
                for item in reader.read(ip) {
                    assert_eq!(item.get_data_slice(), &bigdata[count..count + item.len]);
                    count += item.len;
                }
                assert_eq!(count, bigdata.len());
            }
        }
    }
}
