use std::{f32::consts::PI, fs::read_dir, io, path::Path};

use clap::Parser;
use hound::WavReader;
use rustfft::{FftPlanner, num_complex::Complex};

#[derive(Parser)]
struct Args {
    /// Url or path for Urban8k data
    input: String,
    /// Path to output data
    output: String,
    /// Minimum frequency
    #[clap(long, default_value_t = 2048)]
    min_freq: usize,
    /// Maximum frequency
    #[clap(long, default_value_t = 8192)]
    max_freq: usize,
    /// Number of STFT windows
    #[clap(long, default_value_t = 168)]
    num_windows: usize,
    /// Clip duration in seconds
    #[clap(long, default_value_t = 4.)]
    clip_duration: f32,
    /// Size of frequency bins
    #[clap(long, default_value_t = 128)]
    freq_bin: usize,
}

/// Generates a spectrogram from a WAV file with optional downsampling.
///
/// # Arguments
/// * `file_path` - Path to the WAV file.
/// * `clip_duration_secs` - Duration of the clip to analyze (in seconds).
/// * `num_time_buckets` - Number of horizontal (time) segments in the spectrogram.
/// * `freq_bin_size` - Number of frequency bins (vertical resolution).
/// * `target_sample_rate` - Desired sample rate for processing.
///
/// # Returns
/// A 2D vector of spectrogram magnitudes [time][frequency].
pub fn generate_spectrogram<R: io::Read>(
    reader: &mut WavReader<R>,
    clip_duration_secs: f32,
    num_time_buckets: usize,
    freq_bin_size: usize,
    target_sample_rate: usize,
) -> Result<Box<[Box<[f32]>]>, Box<dyn std::error::Error>> {
    // Read the WAV file
    let spec = reader.spec();
    let original_sample_rate = spec.sample_rate as usize;

    // Read samples as f32
    let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32).collect();

    // Resample if needed
    let resampled = if original_sample_rate != target_sample_rate {
        resample_linear(&samples, original_sample_rate, target_sample_rate)
    } else {
        samples
    };

    let sample_rate = target_sample_rate;
    let num_samples = (clip_duration_secs * sample_rate as f32) as usize;
    let resampled = &resampled[..num_samples.min(resampled.len())];

    // Compute window size
    let window_size = num_samples / num_time_buckets;
    let fft_size = freq_bin_size * 2;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let hann_window: Vec<f32> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos()))
        .collect();

    let mut spectrogram = Vec::with_capacity(num_time_buckets).into_boxed_slice();

    for (i, bucket) in spectrogram.iter_mut().enumerate() {
        let start = i * window_size;
        let end = start + fft_size;

        if end > resampled.len() {
            break;
        }

        let mut buffer: Vec<Complex<f32>> = resampled[start..end]
            .iter()
            .zip(&hann_window)
            .map(|(s, w)| Complex::new(s * w, 0.0))
            .collect();

        fft.process(&mut buffer);

        *bucket = buffer[..freq_bin_size].iter().map(|c| c.norm()).collect();
    }

    Ok(spectrogram)
}

/// Simple linear resampling from one sample rate to another.
fn resample_linear(samples: &[f32], original_rate: usize, target_rate: usize) -> Vec<f32> {
    let resample_ratio = target_rate as f32 / original_rate as f32;
    let new_len = (samples.len() as f32 * resample_ratio).floor() as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let interp_pos = i as f32 / resample_ratio;
        let low = interp_pos.floor() as usize;
        let high = interp_pos.ceil() as usize;

        if high >= samples.len() {
            break;
        }

        let weight = interp_pos - low as f32;
        let value = samples[low] * (1.0 - weight) + samples[high] * weight;
        resampled.push(value);
    }

    resampled
}

fn main() {
    let args = Args::parse();
    let input_path = Path::new(&args.input);
    if !input_path.exists() {
        println!("Input path does not exist. Please provide a valid path.");
        return;
    }
    if !input_path.is_dir() {
        println!("Input path is not a directory. Please provide a valid directory.");
        return;
    }
    let input_dir = read_dir(input_path).expect("Input directory should be valid");
    let output_path = Path::new(&args.output);
    if output_path.exists() {
        println!("Output directory already exists. Please delete it or choose a different name.");
        return;
    }
    std::fs::create_dir_all(output_path).expect("Failed to create output directory");

    // take the minum samplerate and multiply by 2 until we reach the max
    let mut sample_rates = Vec::new();
    let mut sample_rate = args.min_freq;
    while sample_rate <= args.max_freq {
        sample_rates.push(sample_rate);
        sample_rate *= 2;
    }
    let sample_rates = sample_rates.into_boxed_slice();

    for entry in input_dir {
        // load wav using hound
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wav") {
            println!("Skipping non-wav file: {:?}", path);
            continue;
        }
        let mut reader = WavReader::open(&path).expect("Failed to open WAV file");
        for &sample_rate in &sample_rates {
            let stft = generate_spectrogram(
                &mut reader,
                args.clip_duration,
                args.num_windows,
                args.freq_bin,
                sample_rate,
            );
            println!("Generated STFT for file: {:?}", stft);
        }
    }
}
