use crate::data::{GeneRecord, CANCER_TYPES, NUM_CN_SIGNATURES};
use anyhow::{Context, Result};
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use tracing::{debug, info, warn};

/// Supported file formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileFormat {
    Csv,
    Tsv,
    GzippedCsv,
    GzippedTsv,
}

impl FileFormat {
    /// Detect file format from path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let ext = path.extension().and_then(|e| e.to_str());
        let stem = path.file_stem().and_then(|s| s.to_str());
        
        match (ext, stem) {
            (Some("gz"), Some(stem)) => {
                if stem.ends_with(".csv") {
                    Ok(FileFormat::GzippedCsv)
                } else if stem.ends_with(".tsv") || stem.ends_with(".txt") {
                    Ok(FileFormat::GzippedTsv)
                } else {
                    Err(anyhow::anyhow!("Cannot determine format of gzipped file"))
                }
            }
            (Some("csv"), _) => Ok(FileFormat::Csv),
            (Some("tsv"), _) | (Some("txt"), _) => Ok(FileFormat::Tsv),
            _ => Err(anyhow::anyhow!("Unsupported file format")),
        }
    }
    
    /// Get delimiter character
    pub fn delimiter(&self) -> u8 {
        match self {
            FileFormat::Csv | FileFormat::GzippedCsv => b',',
            FileFormat::Tsv | FileFormat::GzippedTsv => b'\t',
        }
    }
    
    /// Check if format is gzipped
    pub fn is_gzipped(&self) -> bool {
        matches!(self, FileFormat::GzippedCsv | FileFormat::GzippedTsv)
    }
}

/// Data loader configuration
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Whether file has header
    pub has_header: bool,
    /// Batch size for reading
    pub batch_size: usize,
    /// Maximum number of records to load (0 = unlimited)
    pub max_records: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            has_header: true,
            batch_size: 10000,
            max_records: 0,
        }
    }
}

/// Data loader for gene records
pub struct DataLoader {
    config: LoaderConfig,
}

impl DataLoader {
    /// Create new data loader with default config
    pub fn new() -> Self {
        Self {
            config: LoaderConfig::default(),
        }
    }
    
    /// Create new data loader with custom config
    pub fn with_config(config: LoaderConfig) -> Self {
        Self { config }
    }
    
    /// Load gene records from file
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Vec<GeneRecord>> {
        let path = path.as_ref();
        info!("Loading data from {:?}", path);
        
        let format = FileFormat::from_path(path)?;
        debug!("Detected file format: {:?}", format);
        
        let records = if format.is_gzipped() {
            self.load_gzipped(path, format)?
        } else {
            self.load_plain(path, format)?
        };
        
        info!("Loaded {} records", records.len());
        Ok(records)
    }
    
    /// Load from plain file
    fn load_plain<P: AsRef<Path>>(&self, path: P, format: FileFormat) -> Result<Vec<GeneRecord>> {
        let file = File::open(path).context("Failed to open file")?;
        let reader = BufReader::new(file);
        self.parse_records(reader, format)
    }
    
    /// Load from gzipped file
    fn load_gzipped<P: AsRef<Path>>(&self, path: P, format: FileFormat) -> Result<Vec<GeneRecord>> {
        let file = File::open(path).context("Failed to open gzipped file")?;
        let gz = GzDecoder::new(file);
        let reader = BufReader::new(gz);
        self.parse_records(reader, format)
    }
    
    /// Parse records from reader
    fn parse_records<R: Read>(&self, reader: R, format: FileFormat) -> Result<Vec<GeneRecord>> {
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(format.delimiter())
            .has_headers(self.config.has_header)
            .from_reader(reader);
        
        let headers = if self.config.has_header {
            csv_reader.headers()?.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        
        debug!("Headers: {:?}", headers);
        
        let mut records = Vec::new();
        let mut record_count = 0;
        
        for result in csv_reader.records() {
            let record = result.context("Failed to parse CSV record")?;
            
            match self.parse_gene_record(&record, &headers) {
                Ok(gene_record) => {
                    records.push(gene_record);
                    record_count += 1;
                    
                    if self.config.max_records > 0 && record_count >= self.config.max_records {
                        warn!("Reached maximum record limit: {}", self.config.max_records);
                        break;
                    }
                    
                    if record_count % self.config.batch_size == 0 {
                        debug!("Loaded {} records...", record_count);
                    }
                }
                Err(e) => {
                    warn!("Failed to parse record at line {}: {}", record_count + 2, e);
                }
            }
        }
        
        Ok(records)
    }
    
    /// Parse a single gene record from CSV record
    fn parse_gene_record(
        &self,
        record: &csv::StringRecord,
        headers: &[String],
    ) -> Result<GeneRecord> {
        let get_field = |name: &str| -> Option<String> {
            headers.iter().position(|h| h == name)
                .and_then(|idx| record.get(idx).map(|s| s.to_string()))
        };
        
        let get_f32 = |name: &str| -> Option<f32> {
            get_field(name)
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|&v| !v.is_nan() && !v.is_infinite())
        };
        
        let sample = get_field("sample")
            .context("Missing required field: sample")?;
        let gene_id = get_field("gene_id")
            .context("Missing required field: gene_id")?;
        
        let mut gene_record = GeneRecord::new(sample, gene_id);
        
        // Parse base features
        gene_record.seg_val = get_f32("segVal");
        gene_record.minor_cn = get_f32("minor_cn");
        gene_record.intersect_ratio = get_f32("intersect_ratio");
        gene_record.purity = get_f32("purity");
        gene_record.ploidy = get_f32("ploidy");
        gene_record.a_score = get_f32("AScore");
        gene_record.p_loh = get_f32("pLOH");
        gene_record.cna_burden = get_f32("cna_burden");
        
        // Parse CN signatures (CN1-CN19)
        for i in 0..NUM_CN_SIGNATURES {
            let field_name = format!("CN{}", i + 1);
            gene_record.cn_signatures[i] = get_f32(&field_name);
        }
        
        // Parse clinical features
        gene_record.age = get_f32("age");
        gene_record.gender = get_f32("gender");
        
        // Parse cancer type from one-hot encoding or direct field
        gene_record.cancer_type = get_field("cancer_type").or_else(|| {
            // Try to infer from one-hot encoded columns
            for (i, &cancer_type) in CANCER_TYPES.iter().enumerate() {
                let field_name = format!("type_{}", cancer_type);
                if let Some(val) = get_f32(&field_name) {
                    if val > 0.5 {
                        return Some(cancer_type.to_string());
                    }
                }
            }
            None
        });
        
        // Parse prior frequencies
        gene_record.freq_linear = get_f32("freq_Linear");
        gene_record.freq_bfb = get_f32("freq_BFB");
        gene_record.freq_circular = get_f32("freq_Circular");
        gene_record.freq_hr = get_f32("freq_HR");
        
        // Parse target
        gene_record.y = get_field("y")
            .and_then(|s| s.parse::<u8>().ok())
            .filter(|&v| v == 0 || v == 1);
        
        Ok(gene_record)
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Load gene frequency data from file
pub fn load_gene_frequencies<P: AsRef<Path>>(path: P) -> Result<std::collections::HashMap<String, [f32; 4]>> {
    let mut frequencies = std::collections::HashMap::new();
    
    let file = File::open(path).context("Failed to open gene frequencies file")?;
    let reader = BufReader::new(file);
    
    for line in reader.lines() {
        let line = line.context("Failed to read line")?;
        let parts: Vec<&str> = line.split('\t').collect();
        
        if parts.len() >= 5 {
            let gene_id = parts[0].to_string();
            let freq_linear = parts[1].parse::<f32>().unwrap_or(0.0);
            let freq_bfb = parts[2].parse::<f32>().unwrap_or(0.0);
            let freq_circular = parts[3].parse::<f32>().unwrap_or(0.0);
            let freq_hr = parts[4].parse::<f32>().unwrap_or(0.0);
            
            frequencies.insert(gene_id, [freq_linear, freq_bfb, freq_circular, freq_hr]);
        }
    }
    
    info!("Loaded {} gene frequency records", frequencies.len());
    Ok(frequencies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    
    #[test]
    fn test_file_format_detection() {
        assert_eq!(FileFormat::from_path("data.csv").unwrap(), FileFormat::Csv);
        assert_eq!(FileFormat::from_path("data.tsv").unwrap(), FileFormat::Tsv);
        assert_eq!(FileFormat::from_path("data.csv.gz").unwrap(), FileFormat::GzippedCsv);
        assert_eq!(FileFormat::from_path("data.tsv.gz").unwrap(), FileFormat::GzippedTsv);
    }
    
    #[test]
    fn test_parse_simple_csv() {
        let csv_data = "sample,gene_id,segVal,y\nS1,G1,2.5,1\nS1,G2,1.0,0";
        let cursor = Cursor::new(csv_data);
        
        let loader = DataLoader::new();
        let records = loader.parse_records(cursor, FileFormat::Csv).unwrap();
        
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].sample, "S1");
        assert_eq!(records[0].gene_id, "G1");
        assert_eq!(records[0].seg_val, Some(2.5));
        assert_eq!(records[0].y, Some(1));
    }
}