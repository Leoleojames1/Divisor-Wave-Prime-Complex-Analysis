/*
 * Sin Wave Sieve Animation
 * by Leo Borcherding.  
 * 
 * Render a set of many sin waves. 
 */

//import gifAnimation.*;
//GifMaker gifExport;

import ddf.minim.*;
import ddf.minim.ugens.*;

Minim minim;
AudioOutput out;
Oscil[] oscillators = new Oscil[100];

Oscil[] plucks = new Oscil[100];
Delay[] delays = new Delay[100];

int xspacing = 4;   // How far apart should each horizontal location be spaced
int waveWidth;              // Width of entire wave
float theta = 0.0;  // Start angle at 0

//float amplitude = 200.0;  // Height of wave
//float period = 500;  // How many pixels before the wave repeats

//float amplitude = 250.0;  // Height of wave
//float period = 300;  // How many pixels before the wave repeats
float amplitude = 225.0;  // Height of wave
float period = 85;  // How many pixels before the wave repeats

float dx;  // Value for incrementing X, a function of period and xspacing
float[] yvalues;  // Using an array to store height values for the wave

// ======================================================================================================
void setup() {
  //Canvas Parameters
  //size(3840, 1080);
  size(1920, 1080);
  frameRate(75); // Set the frame rate to 12 fps
  strokeWeight(1);
  waveWidth = width+4;
  dx = (TWO_PI / period) * xspacing;
  yvalues = new float[waveWidth/xspacing];
  //minim = new Minim(this);
  //out = minim.getLineOut(Minim.STEREO, 2048);
  delay(5000);
  //for (int i = 0; i < oscillators.length; i++) {
  //  int divisor2 = i + 2; // Define divisor based on i
  //  float frequency = PI * divisor2; // Frequency is a multiple of PI
  //  oscillators[i] = new Oscil(frequency, 0.001f, Waves.SINE);
  //  oscillators[i].patch(out);
    
  //  // Initialize the Oscil and Delay objects
  //  plucks[i] = new Oscil( 440, 0.5f, Waves.SINE);
  //  delays[i] = new Delay( 0.4, 0.5, true, true );
  //  plucks[i].patch(delays[i]).patch(out);
  //}
    
  //gifExport = new GifMaker(this, "out.gif");
  //gifExport.setRepeat(0); // 0 means "loop forever"
  //gifExport.addFrame();
  //gifExport.setDelay(1000/12); // Set the delay to 1000/12 ms, which means 12 fps
}

// ======================================================================================================
void draw() {
  //Method Calls
  background(0);
  for (int divisor = 1; divisor <= oscillators.length; divisor++) {
  //for (int divisor = 2; divisor <= 34; divisor++) {
    calcWave(divisor);
    renderWave(divisor);
    //convertWave2Sound(yvalues, divisor);
  }
  //gifExport.addFrame(); // Add this line here
}

// ======================================================================================================
void calcWave(int divisor) {
  // Increment theta (try different values for 'angular velocity' here
  //theta += 0.05;
  theta += 0.025;
  //theta += 0.0125;
  //theta += 0.006125;
  //theta += 0.001;
  
  //period += 0.1;
  // For every x value, calculate a y value with sine function
  float x = theta;
  for (int i = 0; i < yvalues.length; i++) {
    yvalues[i] = -sin(x/divisor)*amplitude; // Add the divisor back to the sine function
    x += dx;
  }
}

// ======================================================================================================
void renderWave(float divisor) {
  stroke(60 + (21%divisor)*8, 40 + (33%divisor)*8, 40 + (43%divisor)*8); // Set the color of the line
  noFill(); // Disable filling
  blendMode(ADD); // Enable additive blending

  for (int x = 0; x < yvalues.length-1; x++) {
    // Draw a line between each pair of points
    line(x*xspacing + 10, height/2+yvalues[x], (x+1)*xspacing + 10, height/2+yvalues[x+1]);
  }

  blendMode(BLEND); // Reset to default blending mode
}
// ======================================================================================================
void convertWave2Sound(float[] yvalues, int divisor) {
  for (int i = 1; i < yvalues.length; i++) { // Start from 1 to avoid out-of-bounds error
  //for (int i = 1; i < divisor; i++) { // Start from 1 to avoid out-of-bounds error
    float frequency = PI * divisor; // Frequency is a multiple of PI
    float distanceFromZero = abs((1-abs(yvalues[i]))); // Calculate the absolute distance of y from 1
    float soundAmplitude = distanceFromZero; // The amplitude is highest when y = 0 and lowest when y = 1 or -1

    // If the sine wave crosses the x-axis, generate a pluck sound with delay
    if (yvalues[i] * yvalues[i-1] < 0) { // Check if the sign has changed
      plucks[divisor - 2].setFrequency(frequency);
      plucks[divisor - 2].setAmplitude(soundAmplitude);
      plucks[divisor - 2].unpatch(out);
      plucks[divisor - 2].patch(delays[divisor - 2]).patch(out);
    }

    oscillators[divisor - 2].setFrequency(frequency);
    oscillators[divisor - 2].setAmplitude(soundAmplitude);
  }
}
