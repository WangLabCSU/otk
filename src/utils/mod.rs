use std::path::Path;
use anyhow::{Context, Result};

/// Ensure directory exists
pub fn ensure_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        std::fs::create_dir_all(path)
            .with_context(|| format!("Failed to create directory: {:?}", path))?;
    }
    Ok(())
}

/// Check if file exists
pub fn file_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_file()
}

/// Get file extension
pub fn get_extension<P: AsRef<Path>>(path: P) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
}

/// Format duration as human-readable string
pub fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:.1}m", secs / 60.0)
    } else {
        format!("{:.1}h", secs / 3600.0)
    }
}

/// Format number with commas
pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;
    
    for c in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
        count += 1;
    }
    
    result.chars().rev().collect()
}

/// Memory usage utilities
pub mod memory {
    /// Format bytes as human-readable string
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_idx = 0;
        
        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }
        
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
    
    /// Get current memory usage (platform-specific)
    #[cfg(target_os = "linux")]
    pub fn current_usage() -> Option<usize> {
        use std::fs;
        
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<usize>().ok().map(|kb| kb * 1024);
                    }
                }
            }
        }
        None
    }
    
    #[cfg(not(target_os = "linux"))]
    pub fn current_usage() -> Option<usize> {
        None
    }
}

/// Random number utilities
pub mod random {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    
    /// Create RNG with fixed seed
    pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }
    
    /// Generate random vector
    pub fn random_vector(size: usize, seed: u64) -> Vec<f32> {
        use rand::Rng;
        
        let mut rng = seeded_rng(seed);
        (0..size).map(|_| rng.gen::<f32>()).collect()
    }
}

/// Validation utilities
pub mod validation {
    use anyhow::{bail, Result};
    
    /// Validate that value is in range
    pub fn in_range<T: PartialOrd>(value: T, min: T, max: T, name: &str) -> Result<()> {
        if value < min || value > max {
            bail!("{} must be between {} and {}, got {}", name, min, max, value);
        }
        Ok(())
    }
    
    /// Validate that value is positive
    pub fn positive<T: PartialOrd + Default>(value: T, name: &str) -> Result<()> {
        if value <= T::default() {
            bail!("{} must be positive, got {}", name, value);
        }
        Ok(())
    }
    
    /// Validate file extension
    pub fn file_extension<P: AsRef<std::path::Path>>(
        path: P,
        allowed: &[&str],
    ) -> Result<()> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        if !allowed.contains(&ext.as_str()) {
            bail!(
                "Invalid file extension: {}. Allowed: {:?}",
                ext,
                allowed
            );
        }
        Ok(())
    }
}

/// Collection utilities
pub mod collections {
    /// Chunk vector into batches
    pub fn chunk<T: Clone>(data: &[T], batch_size: usize) -> Vec<Vec<T>> {
        data.chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    /// Get unique elements
    pub fn unique<T: Clone + Eq + std::hash::Hash>(data: &[T]) -> Vec<T> {
        use std::collections::HashSet;
        
        let set: HashSet<_> = data.iter().cloned().collect();
        set.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30.0s");
        assert_eq!(format_duration(90.0), "1.5m");
        assert_eq!(format_duration(3600.0), "1.0h");
    }
    
    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(1234567), "1,234,567");
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(memory::format_bytes(1024), "1.00 KB");
        assert_eq!(memory::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(memory::format_bytes(1536), "1.50 KB");
    }
    
    #[test]
    fn test_random_vector() {
        let v1 = random::random_vector(10, 42);
        let v2 = random::random_vector(10, 42);
        let v3 = random::random_vector(10, 43);
        
        assert_eq!(v1.len(), 10);
        assert_eq!(v1, v2); // Same seed = same values
        assert_ne!(v1, v3); // Different seed = different values
    }
    
    #[test]
    fn test_validation() {
        assert!(validation::in_range(0.5, 0.0, 1.0, "value").is_ok());
        assert!(validation::in_range(1.5, 0.0, 1.0, "value").is_err());
        
        assert!(validation::positive(1.0, "value").is_ok());
        assert!(validation::positive(0.0, "value").is_err());
    }
    
    #[test]
    fn test_collections() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let chunks = collections::chunk(&data, 3);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[3], vec![10]);
        
        let data = vec![1, 2, 2, 3, 3, 3];
        let unique_vals = collections::unique(&data);
        assert_eq!(unique_vals.len(), 3);
    }
}