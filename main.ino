#include "esp_dsp.h"
#include <Adafruit_LSM6DSOX.h>
#include <Wire.h>
#include <TheAnton205-project-1_inferencing.h> 

Adafruit_LSM6DSOX sox;

const int N = 256;
float* samples_real;
float* samples_imag;
float* window;
float* complex_data;

void setup() {
  Serial.begin(115200);
  Wire.begin(8, 9);

  if (!sox.begin_I2C(0x6A, &Wire)) {
    Serial.println("ERROR: LSM6DSOX sensor not found");
    while (1) delay(10);
  }
  sox.setAccelDataRate(LSM6DS_RATE_104_HZ);

  // 16-byte aligned heap allocation for S3 vector DSP instructions
  samples_real = (float*)heap_caps_malloc(N * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  samples_imag = (float*)heap_caps_malloc(N * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  window       = (float*)heap_caps_malloc(N * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  complex_data = (float*)heap_caps_malloc(N * 2 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  dsps_fft2r_init_fc32(NULL, 1024);
  dsps_wind_hann_f32(window, N);

  Serial.println("Edge Impulse Predictive Maintenance Node Initialized.");
  Serial.println("Awaiting structural data...");
}

void loop() {
  // collect samples
  for (int i = 0; i < N; i++) {
    sensors_event_t accel, gyro, temp;
    sox.getEvent(&accel, &gyro, &temp);
    samples_real[i] = accel.acceleration.z;
    samples_imag[i] = 0;
  }

  // remove dc offset
  float mean = 0;
  for (int i = 0; i < N; i++) mean += samples_real[i];
  mean /= N;
  for (int i = 0; i < N; i++) samples_real[i] -= mean;

  // hann window
  dsps_mul_f32(samples_real, window, samples_real, N, 1, 1, 1);

  // pack real + imag
  for (int i = 0; i < N; i++) {
    complex_data[i * 2]     = samples_real[i];
    complex_data[i * 2 + 1] = samples_imag[i];
  }

  // run esp32 fft
  dsps_fft2r_fc32(complex_data, N);
  dsps_bit_rev_fc32(complex_data, N);

  // calculate band energies
  float infra = 0, low = 0, mid = 0, high = 0;
  float peak_mag = 0;
  int   peak_bin = 0;

  for (int i = 1; i < N / 2; i++) {
    float re  = complex_data[i * 2];
    float im  = complex_data[i * 2 + 1];
    float mag = sqrt(re * re + im * im);

    if (i <= 4)        infra += mag;
    else if (i <= 24)  low   += mag;
    else if (i <= 49)  mid   += mag;
    else if (i <= 100) high  += mag;

    if (mag > peak_mag) {
      peak_mag = mag;
      peak_bin = i;
    }
  }

  float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {
      infra, 
      low, 
      mid, 
      high
  };

  // wrap array for edge impulse
  signal_t signal;
  int err = numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) {
      Serial.println("ERR: Failed to create signal from buffer");
      return;
  }

  // run classifier
  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false); 
  
  if (res != EI_IMPULSE_OK) {
      Serial.printf("ERR: Failed to run classifier (%d)\n", res);
      return;
  }

  // print predictions
  Serial.println("=== AI STRUCTURAL HEALTH PREDICTION ===");
  for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
      Serial.printf("  %s: %.2f%%\n", result.classification[i].label, result.classification[i].value * 100.0);
  }
  
  // #if EI_CLASSIFIER_HAS_ANOMALY == 1
  // Serial.printf("  Anomaly Score: %.3f\n", result.anomaly);
  // #endif
  
  Serial.println("=======================================\n");

  delay(10); // Small loop delay
}