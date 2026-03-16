// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Routines for encoding and decoding miniblock data
//!
//! Miniblock encoding is one of the two structural encodings in Lance 2.1.
//! In this approach the data is compressed into a series of chunks put into
//! a single buffer.
//!
//! A chunk must be encoded or decoded as a unit.  There is a small amount of
//! chunk metadata such as the number and size of each buffer in the chunk.
//!
//! Any form of compression can be used since we are compressing and decompressing
//! entire chunks.
use crate::{
    buffer::LanceBuffer, data::DataBlock, format::pb21::CompressiveEncoding,
    version::LanceFileVersion,
};

use lance_core::{Error, Result};

/// Default pre-serialization byte budget for miniblock value buffers.
///
/// This remains the default when users do not specify any miniblock tuning metadata.
pub const MAX_MINIBLOCK_BYTES: u64 = 8 * 1024 - 6;
/// Default maximum number of values per miniblock chunk.
///
/// This remains the default when users do not specify any miniblock tuning metadata.
pub const MAX_MINIBLOCK_VALUES: u64 = 4096;
/// Maximum miniblock values supported by the current metadata layout in Lance 2.2+.
const MAX_MINIBLOCK_VALUES_V2_2: u64 = 16 * 1024;
/// Maximum serialized miniblock size in Lance 2.1 and earlier.
const MAX_MINIBLOCK_SERIALIZED_BYTES_V2_1: u64 = 32 * 1024;
/// Maximum serialized miniblock size in Lance 2.2+.
///
/// The metadata stores the chunk size in 28 bits of 8-byte words.
const MAX_MINIBLOCK_SERIALIZED_BYTES_V2_2: u64 = (1u64 << 28) * 8;
/// Maximum rep/def buffer size that can be represented in the miniblock header.
pub(super) const MAX_MINIBLOCK_LEVEL_BUFFER_BYTES: u64 = u16::MAX as u64;

/// Resolved miniblock tuning limits for a single field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MiniBlockLimits {
    pub max_bytes: u64,
    pub max_values: u64,
}

impl Default for MiniBlockLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_MINIBLOCK_BYTES,
            max_values: MAX_MINIBLOCK_VALUES,
        }
    }
}

impl MiniBlockLimits {
    pub fn try_new(
        version: LanceFileVersion,
        max_values: Option<u64>,
        max_bytes: Option<u64>,
    ) -> Result<Self> {
        let max_values = max_values.unwrap_or(MAX_MINIBLOCK_VALUES);
        let max_bytes = max_bytes.unwrap_or(MAX_MINIBLOCK_BYTES);
        if max_values < 2 {
            return Err(Error::invalid_input(format!(
                "miniblock-max-values must be >= 2, got {max_values}"
            )));
        }
        if max_values > Self::max_values_for_version(version) {
            return Err(Error::invalid_input(format!(
                "miniblock-max-values {max_values} exceeds the limit {} for file version {version}",
                Self::max_values_for_version(version)
            )));
        }
        if max_bytes == 0 {
            return Err(Error::invalid_input(
                "miniblock-max-bytes must be > 0".to_string(),
            ));
        }
        Ok(Self {
            max_bytes,
            max_values,
        })
    }

    pub fn max_non_last_chunk_values(self) -> u64 {
        1u64 << (u64::BITS - 1 - self.max_values.leading_zeros())
    }

    pub fn with_byte_ceiling(self, max_bytes: u64) -> Result<Self> {
        if max_bytes == 0 {
            return Err(Error::invalid_input(
                "miniblock-max-bytes must be > 0".to_string(),
            ));
        }
        Ok(Self {
            max_bytes: self.max_bytes.min(max_bytes),
            ..self
        })
    }

    pub fn with_minichunk_size(self, minichunk_size: Option<i64>) -> Result<Self> {
        match minichunk_size {
            Some(size) if size > 0 => self.with_byte_ceiling(size as u64),
            _ => Ok(self),
        }
    }

    pub fn max_values_for_version(version: LanceFileVersion) -> u64 {
        if version >= LanceFileVersion::V2_2 {
            MAX_MINIBLOCK_VALUES_V2_2
        } else {
            MAX_MINIBLOCK_VALUES
        }
    }

    pub fn max_log_num_values(version: LanceFileVersion) -> u8 {
        if version >= LanceFileVersion::V2_2 {
            14
        } else {
            12
        }
    }

    pub fn max_serialized_chunk_bytes(version: LanceFileVersion) -> u64 {
        if version >= LanceFileVersion::V2_2 {
            MAX_MINIBLOCK_SERIALIZED_BYTES_V2_2
        } else {
            MAX_MINIBLOCK_SERIALIZED_BYTES_V2_1
        }
    }
}

/// Page data that has been compressed into a series of chunks put into
/// a single buffer.
#[derive(Debug)]
pub struct MiniBlockCompressed {
    /// The buffers of compressed data
    pub data: Vec<LanceBuffer>,
    /// Describes the size of each chunk
    pub chunks: Vec<MiniBlockChunk>,
    /// The number of values in the entire page
    pub num_values: u64,
}

/// Describes the size of a mini-block chunk of data
///
/// Mini-block chunks are designed to be small (just a few disk sectors)
/// and contain a power-of-two number of values (except for the last chunk)
///
/// By default we limit a chunk to 4Ki values and slightly less than
/// 8KiB of compressed value data. These defaults can be lowered or raised
/// through field metadata as long as the resulting chunks remain valid for
/// the selected file version.
#[derive(Debug)]
pub struct MiniBlockChunk {
    // The size in bytes of each buffer in the chunk.
    //
    // In Lance 2.1, the chunk size is limited to 32KiB, so only 16-bits are used.
    // Since Lance 2.2, the chunk size uses u32 to support larger chunk size
    pub buffer_sizes: Vec<u32>,
    // The log (base 2) of the number of values in the chunk.  If this is the final chunk
    // then this should be 0 (the number of values will be calculated by subtracting the
    // size of all other chunks from the total size of the page)
    //
    // For example, 1 would mean there are 2 values in the chunk and 12 would mean there
    // are 4Ki values in the chunk.
    pub log_num_values: u8,
}

impl MiniBlockChunk {
    /// Gets the number of values in this block
    ///
    /// This requires `vals_in_prev_blocks` and `total_num_values` because the
    /// last block in a page is a special case which stores 0 for log_num_values
    /// and, in that case, the number of values is determined by subtracting
    /// `vals_in_prev_blocks` from `total_num_values`
    pub fn num_values(&self, vals_in_prev_blocks: u64, total_num_values: u64) -> u64 {
        if self.log_num_values == 0 {
            total_num_values - vals_in_prev_blocks
        } else {
            1 << self.log_num_values
        }
    }
}

/// Trait for compression algorithms that are suitable for use in the miniblock structural encoding
///
/// These compression algorithms should be capable of encoding the data into small chunks
/// where each chunk (except the last) has 2^N values (N can vary between chunks)
pub trait MiniBlockCompressor: std::fmt::Debug + Send + Sync {
    /// Compress a `page` of data into multiple chunks
    ///
    /// See [`MiniBlockCompressed`] for details on how chunks should be sized.
    ///
    /// This method also returns a description of the encoding applied that will be
    /// used at decode time to read the data.
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, CompressiveEncoding)>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miniblock_limits_default_values() {
        let limits = MiniBlockLimits::default();
        assert_eq!(limits.max_values, MAX_MINIBLOCK_VALUES);
        assert_eq!(limits.max_bytes, MAX_MINIBLOCK_BYTES);
        assert_eq!(limits.max_non_last_chunk_values(), MAX_MINIBLOCK_VALUES);
    }

    #[test]
    fn test_miniblock_limits_v2_1_caps_values() {
        let error = MiniBlockLimits::try_new(LanceFileVersion::V2_1, Some(8192), None)
            .expect_err("v2.1 should reject values above 4096");
        assert!(
            error
                .to_string()
                .contains("miniblock-max-values 8192 exceeds the limit 4096")
        );
    }

    #[test]
    fn test_miniblock_limits_v2_2_allows_larger_value_cap() {
        let limits = MiniBlockLimits::try_new(LanceFileVersion::V2_2, Some(12_000), None)
            .expect("v2.2 should allow values above 4096");
        assert_eq!(limits.max_values, 12_000);
        assert_eq!(limits.max_non_last_chunk_values(), 8192);
    }
}
