#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

union DoubleToInt {
  double dVal;
  uint64_t iVal;
};


void fastTwoSum(float a, float b, float *s, float *t)
{
  float temp;

  // calculate the result
  *s = a + b;
  // calculate the error in the sum
  temp = *s - a;
  *t = b - temp;
}


void writeToCSV(const double *avgs, const char *filename, const long int K)
{
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    printf("Error opening file");
    return;
  }

  // iterate through the array
  for (int i = 0; i<K/1000; i++) {
    fprintf(file, "%.30f", avgs[i]);
    // print a comma if not end of file
    if (i < K/1000 - 1)
      fprintf(file, ", ");
  }
  fclose(file);
}

void writeHarmonicError(const double *errors, const int K, FILE *file)
{

  // write each of the lists to csv independently
  for (int i = 0; i<K/1000000; i++) {
    fprintf(file, "%.30f", errors[i]);
    // print a comma if not end of file
    if (i < K/1000000 - 1)
      fprintf(file, ", ");
  }
  fprintf(file, "\n"); // print newline

}

/*
  A function that rounds a binary64 value to a binary32 value
  stochastically. Implemented by treating FP number representations
  as integer values.
*/
float SR(double x) {

  union DoubleToInt temp;
  temp.dVal = x;
  uint32_t r = rand() & 0x1FFFFFFF;
  temp.iVal += r;
  temp.iVal = temp.iVal & 0xFFFFFFFFE0000000;

  return (float)temp.dVal;
}


/* --------------------------------- */
/*              PART 1               */
/* --------------------------------- */
  
// Implement SR_alternative according to the Eqn 1.
float SR_alternative(double x) {

  // calculate RA(x) and RZ(x)

  // find the bordering floats, up and down
  float closest = (float)x;
  float down, up, rz, ra;
  if (closest > x) {
    down = nextafterf(closest, -INFINITY);
    up = closest;
  } else {
    down = closest;
    up = nextafterf(closest, INFINITY);
  }
  // pick the round away and round to zero
  if (x < 0) {
    rz = up;
    ra = down;
  } else {
    ra = up;
    rz = down;
  }

  // define P, not using srand() to keep determinism
  double P = (double)rand() / ((double)RAND_MAX + 1.0);
  // calculate p = ((x-RZ(x)) / (RA(x)-RZ(x)))
  double p = (x - rz) / (ra - rz);
  // Choose either RA or RZ
  if (P < p)
    return ra;
  return rz;
}


const long int K = 5000000;

int main() {
  
  // An arbitrary value for rounding.
  double sample = M_PI;
  double avg = 0;
  double avg_alternative = 0;

  // Calculate the neighbouring binary32 values.
  float closest = (float)sample;
  float down, up;
  if (closest > sample) {
    down = nextafterf(closest, -INFINITY);
    up = closest;
  }
  else {
    down = closest;
    up = nextafterf(closest, INFINITY);
  }

  // Round many times, and calculate the average values as well as count
  // the numbers of times rounding was up/down.
  int countup = 0;
  int countdown = 0;
  double sum, alt_sum = 0;
  float sr_result, sr_alt_result;
  double sr_avgs[K/1000], sr_alt_avgs[K/1000];
  for (int i = 1; i <= K; i++) {
    // get the SR floats
    sr_result = SR(sample);
    sr_alt_result = SR_alternative(sample);
    // get the number of times counted up or counted down
    if (sr_alt_result > sample) {
      countdown++;
    } else {
      countup++;
    }
    // add to the average
    avg += sr_result;
    avg_alternative += sr_alt_result;

    // every 1000 steps append the absoluete error of avg result for both SR algs to array
    if (i % 1000 == 0) {
      sr_avgs[i/1000 - 1] = fabs(sample - avg / (double)i);
      sr_alt_avgs[i/1000 - 1] = fabs(sample - avg_alternative / (double)i);
    }
  }
  // avg = sum / (double)K;
  // avg_alternative = alt_sum / (double)K;
  avg /= (double)K;
  avg_alternative /= (double)K;

  // Print out some useful stats.
  printf("Value being rounded:           %.60f \n", sample);
  printf("SR average value:              %.60f \n", avg);
  printf("SR_alternative average value:  %.60f \n", avg_alternative);
  printf("Binary32 value before:         %.60f \n", down);
  printf("Binary32 value after:          %.60f \n", up);
  printf("Closest binary32:              %.60f \n", closest);


  // Print out the average of all rounded values to a file as csv
  writeToCSV(sr_avgs, "results/sr_error.csv", K);
  writeToCSV(sr_alt_avgs, "results/alt_sr_error.csv", K);

  
  // Check that SR_alternative function is correct by comparing the probabilities
  // of rounding up/down, and the expected probability. Print them out
  // below.

  // get error for SR and SR_alternative
  double up_error = fabs(sample - up);
  double down_error = fabs(sample - down);
  double expected_up_prob = up_error / (up - down);
  double expected_down_prob = down_error / (up - down);
  double prob_up = countup / (double)K;
  double prob_down = countdown / (double)K;
  printf("Expected down prob:            %.60f \n", expected_down_prob);
  printf("Down prob:                     %.60f \n", prob_down);
  printf("Expected up prob:              %.60f \n", expected_up_prob);
  printf("Up prob:                       %.60f \n", prob_up);

  /* --------------------------------- */
  /*              PART 2               */
  /* --------------------------------- */

  long int N = 500000000;
  float fharmonic = 0;
  float fharmonic_temp = 0;
  float fharmonic_sr = 0;
  float fharmonic_comp = 0;
  double dharmonic = 0;
  double sr_temp;

  // Error term in the compensated summation.
  float t = 0;
  int b32_stagnated = 0;

  const int array_length = N / 1000000;
  double fharmonic_error[array_length];
  double fharmonic_sr_error[array_length];
  double fharmonic_comp_error[array_length];
  
  for (int i = 1; i <= N; i++) {
    fharmonic_temp = fharmonic;
    // Recursive sum, binary32 RN
    fharmonic += (float)1/i;
    // if fharmonic hasn't changed then it has stagnated
    if (fharmonic == fharmonic_temp && b32_stagnated == 0) {
      printf("Binary32 stagnated at iteration %i \n", i);
      b32_stagnated = 1;
    }
    // Recursive sum, binary32 SR
    sr_temp = (double)fharmonic_sr + (double)1/i;
    fharmonic_sr = SR_alternative(sr_temp);
    // Recursize sum, binary64 RN
    dharmonic += (double)1/i;
    // Other summation methods, TODO.
    // compensated summation
    // calculate addend (current term + previous loss)
    float addend = (float)1/i + t;
    fastTwoSum(fharmonic_comp, addend, &fharmonic_comp, &t);

    // every million iterations calculate absolute errors and save them
    if (i % 1000000 == 0) {
      fharmonic_error[i/1000000 - 1] = fabs(dharmonic - (double)fharmonic);
      fharmonic_sr_error[i/1000000 - 1] = fabs(dharmonic - (double)fharmonic_sr);
      fharmonic_comp_error[i/1000000 - 1] = fabs(dharmonic - (double)fharmonic_comp);
    }
  }

  // write the lists to a csv file
  // open file
  FILE *file = fopen("results/harmonic_errors.csv", "w");
  if (file == NULL) {
    printf("Error opening file");
    return 0;
  }
  // write the errors to the file one by one
  writeHarmonicError(fharmonic_error, N, file);
  writeHarmonicError(fharmonic_sr_error, N, file);
  writeHarmonicError(fharmonic_comp_error, N, file);
  // close the file
  fclose(file);

  printf("Values of the harmonic series after %ld iterations \n", N);
  printf("Recursive summation, binary32:          %.30f \n", fharmonic);
  printf("Recursive summation with SR, binary32:  %.30f \n", fharmonic_sr);
  printf("Compensated summation, binary32:        %.30f \n", fharmonic_comp);
  printf("Recursive summation, binary64:          %.30f \n", dharmonic);

  double b32_abs_error = fabs(dharmonic - (double) fharmonic);
  double sr_abs_error = fabs(dharmonic - (double) fharmonic_sr);
  double comp_abs_error = fabs(dharmonic - (double) fharmonic_comp);
  printf("Recursive summation, binary 32 error:   %.30f \n", b32_abs_error);
  printf("Recursive summation, with SR, error:    %.30f \n", sr_abs_error);
  printf("Compensated sum, binary 32 error:       %.30f \n", comp_abs_error);

  /* --------------------------------- */
  /*              PART 3               */
  /* --------------------------------- */

  // TODO

  return 0;
}
