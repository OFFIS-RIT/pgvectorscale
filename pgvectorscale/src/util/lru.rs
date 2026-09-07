use std::hash::Hash;
use std::num::NonZero;

/// Wrapper around LruCache that tracks statistics about the cache
/// and warns on first eviction using a parameterized message.
use lru::LruCache;
use pgrx::warning;

#[derive(Copy, Clone, Debug, Default)]
pub struct CacheStats {
    pub inserts: usize,
    pub updates: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

impl CacheStats {
    fn record_insert(&mut self, evicted: bool, cache_name: &str, capacity: NonZero<usize>) {
        self.inserts += 1;
        if evicted {
            if self.evictions == 0 {
                warning!(
                    "{} cache is full after processing {} vectors; consider increasing maintenance_work_mem",
                    cache_name,
                    self.inserts
                );
            }
            self.evictions += 1;
            if self.evictions.is_multiple_of(10000) {
                pgrx::debug1!(
                    "{} cache capacity {}, stats: {:?}",
                    cache_name,
                    capacity,
                    self,
                );
            }
        }
    }
}

pub struct LruCacheWithStats<K: Hash + Eq + Clone, V> {
    cache: LruCache<K, V>,
    cache_name: String,
    stats: CacheStats,
}

impl<K: Hash + Eq + Clone, V> LruCacheWithStats<K, V> {
    #[allow(dead_code)]
    pub fn new(capacity: NonZero<usize>, cache_name: &str) -> Self {
        LruCacheWithStats {
            cache: LruCache::new(capacity),
            cache_name: cache_name.to_string(),
            stats: CacheStats::default(),
        }
    }

    /// Bounds entries without reserving the full capacity up front.
    pub fn new_lazy(capacity: NonZero<usize>, cache_name: &str) -> Self {
        let mut cache = LruCache::unbounded();
        cache.resize(capacity);
        LruCacheWithStats {
            cache,
            cache_name: cache_name.to_string(),
            stats: CacheStats::default(),
        }
    }

    pub fn cap(&self) -> NonZero<usize> {
        self.cache.cap()
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Pushes a key-value pair into the cache.
    ///
    /// # Returns
    ///
    /// * `None` if the key was already in the cache (update) or if the cache had space
    /// * `Some((K, V))` if an existing key-value pair was evicted to make space
    ///
    /// # Note
    ///
    /// This differs from the underlying `LruCache::push` method, which returns:
    /// * The old value when updating an existing key
    /// * The evicted key-value pair when inserting a new key
    pub fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        let result = self.cache.push(key.clone(), value);
        if let Some((old_key, _)) = &result {
            if old_key == &key {
                // The key was already in the cache, so we didn't evict anything
                self.stats.updates += 1;
                return None;
            }
        }
        self.stats
            .record_insert(result.is_some(), &self.cache_name, self.cache.cap());
        result
    }

    pub fn get_or_insert<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &V {
        let capacity = self.cache.cap();
        let full = self.cache.len() == capacity.get();
        let mut inserted = false;
        let value = self.cache.get_or_insert(key, || {
            let value = f();
            inserted = true;
            value
        });
        if inserted {
            self.stats.misses += 1;
            self.stats.record_insert(full, &self.cache_name, capacity);
        } else {
            self.stats.hits += 1;
        }
        value
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let result = self.cache.get(key);
        if result.is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        result
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn into_parts(self) -> (LruCache<K, V>, CacheStats) {
        (self.cache, self.stats)
    }

    /// Remove and return the least recently used key-value pair
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        if let Some((key, value)) = self.cache.pop_lru() {
            self.stats.evictions += 1;
            Some((key, value))
        } else {
            None
        }
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;
    use pgrx::pg_test;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::{cell::Cell, rc::Rc};

    #[pg_test]
    fn eager_and_lazy_values_and_lookup_stats_match() {
        let capacity = NonZero::new(2).unwrap();
        for mut cache in [
            LruCacheWithStats::new(capacity, "test"),
            LruCacheWithStats::new_lazy(capacity, "test"),
        ] {
            assert_eq!(cache.cap(), capacity);
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.get_or_insert(1, || vec![10]), &[10]);
            assert_eq!(cache.stats().hits, 0);
            assert_eq!(cache.stats().misses, 1);
            assert_eq!(cache.get_or_insert(1, || panic!("cache hit")), &[10]);
            assert_eq!(cache.get_or_insert(2, || vec![20]), &[20]);
            assert_eq!(cache.get_or_insert(2, || panic!("full cache hit")), &[20]);
            assert_eq!(cache.len(), 2);
            assert_eq!(cache.stats().hits, 2);
            assert_eq!(cache.stats().misses, 2);
            assert_eq!(cache.stats().inserts, 2);
            assert_eq!(cache.stats().updates, 0);
            assert_eq!(cache.stats().evictions, 0);
            assert!(cache.push(1, vec![11]).is_none());
            assert_eq!(cache.get_or_insert(1, || panic!("updated entry")), &[11]);
            assert_eq!(cache.stats().hits, 3);
            assert_eq!(cache.stats().misses, 2);
            assert_eq!(cache.stats().inserts, 2);
            assert_eq!(cache.stats().updates, 1);
            assert_eq!(cache.stats().evictions, 0);
        }
    }

    #[pg_test]
    fn lazy_capacity_does_not_require_upfront_reservation() {
        let capacity = NonZero::new(usize::MAX / 2).unwrap();
        let mut cache = LruCacheWithStats::new_lazy(capacity, "test");
        assert_eq!(cache.cap(), capacity);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get_or_insert(1, || 10), &10);
        assert_eq!(cache.len(), 1);
    }

    #[pg_test]
    fn drop_releases_owned_values() {
        struct Tracked(Rc<Cell<usize>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let capacity = NonZero::new(2).unwrap();
        for mut cache in [
            LruCacheWithStats::new(capacity, "test"),
            LruCacheWithStats::new_lazy(capacity, "test"),
        ] {
            let drops = Rc::new(Cell::new(0));
            cache.get_or_insert(1, || Tracked(Rc::clone(&drops)));
            cache.get_or_insert(2, || Tracked(Rc::clone(&drops)));
            assert_eq!(drops.get(), 0);
            drop(cache);
            assert_eq!(drops.get(), 2);
        }
    }
    // Eviction emits a PostgreSQL warning, so exercise it in a backend.
    #[pgrx::pg_test]
    fn bounded_eviction_preserves_stats_and_recency() {
        let capacity = NonZero::new(2).unwrap();
        for mut cache in [
            LruCacheWithStats::new(capacity, "test"),
            LruCacheWithStats::new_lazy(capacity, "test"),
        ] {
            cache.get_or_insert(1, || vec![10]);
            cache.get_or_insert(2, || vec![20]);
            cache.get_or_insert(1, || panic!("cache hit"));
            assert_eq!(cache.get_or_insert(3, || vec![30]), &[30]);
            assert_eq!(cache.len(), 2);
            assert_eq!(cache.cap(), capacity);
            assert!(cache.cache.contains(&1));
            assert!(!cache.cache.contains(&2));
            assert_eq!(cache.stats().hits, 1);
            assert_eq!(cache.stats().misses, 3);
            assert_eq!(cache.stats().inserts, 3);
            assert_eq!(cache.stats().updates, 0);
            assert_eq!(cache.stats().evictions, 1);
            assert_eq!(cache.push(4, vec![40]), Some((1, vec![10])));
            assert_eq!(cache.len(), 2);
            assert_eq!(cache.stats().inserts, 4);
            assert_eq!(cache.stats().evictions, 2);
            assert_eq!(cache.stats().hits, 1);
            assert_eq!(cache.stats().misses, 3);
        }
    }

    #[pg_test]
    fn eviction_and_replacement_drop_each_owned_value_once() {
        struct Tracked(Rc<Cell<usize>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let capacity = NonZero::new(2).unwrap();
        for mut cache in [
            LruCacheWithStats::new(capacity, "test"),
            LruCacheWithStats::new_lazy(capacity, "test"),
        ] {
            let drops: [_; 5] = std::array::from_fn(|_| Rc::new(Cell::new(0)));
            cache.get_or_insert(1, || Tracked(Rc::clone(&drops[0])));
            cache.get_or_insert(2, || Tracked(Rc::clone(&drops[1])));
            cache.get_or_insert(3, || Tracked(Rc::clone(&drops[2])));
            assert_eq!(drops.each_ref().map(|count| count.get()), [1, 0, 0, 0, 0]);

            assert!(cache.push(3, Tracked(Rc::clone(&drops[3]))).is_none());
            assert_eq!(drops.each_ref().map(|count| count.get()), [1, 0, 1, 0, 0]);
            let evicted = cache.push(4, Tracked(Rc::clone(&drops[4]))).unwrap();
            assert_eq!(evicted.0, 2);
            // push transfers ownership of the evicted value to its caller.
            assert_eq!(drops.each_ref().map(|count| count.get()), [1, 0, 1, 0, 0]);
            drop(evicted);
            assert_eq!(drops.each_ref().map(|count| count.get()), [1, 1, 1, 0, 0]);
            assert_eq!(cache.stats().inserts, 4);
            assert_eq!(cache.stats().updates, 1);
            assert_eq!(cache.stats().evictions, 2);
            drop(cache);
            assert_eq!(drops.each_ref().map(|count| count.get()), [1, 1, 1, 1, 1]);
        }
    }

    #[pg_test]
    fn initializer_panic_preserves_entries_stats_and_reusability() {
        let capacity = NonZero::new(2).unwrap();
        for initial_entries in [1, 2] {
            for mut cache in [
                LruCacheWithStats::new(capacity, "test"),
                LruCacheWithStats::new_lazy(capacity, "test"),
            ] {
                for key in 1..=initial_entries {
                    cache.get_or_insert(key, || vec![key * 10]);
                }
                cache.get_or_insert(1, || panic!("cache hit"));
                let before_stats = *cache.stats();
                let before_entries: Vec<_> = cache
                    .cache
                    .iter()
                    .map(|(key, value)| (*key, value.clone()))
                    .collect();
                // This is a Rust unwind, not a PostgreSQL ERROR/longjmp.
                let result = catch_unwind(AssertUnwindSafe(|| {
                    cache.get_or_insert(3, || panic!("initializer failed"));
                }));
                assert!(result.is_err());
                assert_eq!(cache.cap(), capacity);
                assert_eq!(cache.len(), initial_entries);
                assert_eq!(cache.stats().inserts, before_stats.inserts);
                assert_eq!(cache.stats().updates, before_stats.updates);
                assert_eq!(cache.stats().hits, before_stats.hits);
                assert_eq!(cache.stats().misses, before_stats.misses);
                assert_eq!(cache.stats().evictions, before_stats.evictions);
                let after_entries: Vec<_> = cache
                    .cache
                    .iter()
                    .map(|(key, value)| (*key, value.clone()))
                    .collect();
                assert_eq!(after_entries, before_entries);

                assert_eq!(cache.get_or_insert(3, || vec![30]), &[30]);
                assert_eq!(cache.get_or_insert(1, || panic!("original entry")), &[10]);
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.stats().inserts, before_stats.inserts + 1);
                assert_eq!(cache.stats().updates, before_stats.updates);
                assert_eq!(cache.stats().hits, before_stats.hits + 1);
                assert_eq!(cache.stats().misses, before_stats.misses + 1);
                assert_eq!(cache.stats().evictions, usize::from(initial_entries == 2));
            }
        }
    }
}
