/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "math.h"

#include "occa.hpp"
#include "occa/tools/json.hpp"

#include "visualizer.hpp"

#if OCCA_GL_ENABLED
visualizer vis;
int click;
#endif

std::string deviceInfo;

int width, height;
int Bx, By;

tFloat heightScale;

tFloat currentTime;
tFloat dx, dt;

int stencilRadius;

int mX, mY;

tFloat freq;
tFloat minU, maxU;

std::vector<tFloat> u1, u2;
std::vector<tFloat> xyz;

void run();
void setupMesh();

void setupSolver();
void solve();

#if OCCA_GL_ENABLED
void glRun();
void update();
void drawMesh();
#endif

int main(int argc, char **argv) {
#if OCCA_GL_ENABLED
  vis.setup("OCL Visualizer", argc, argv);
  glRun();
#else
  run();
#endif

  return 0;
}

#if OCCA_GL_ENABLED
void glRun() {
  setupMesh();
  setupSolver();

  click = 0;

  vis.setExternalFunction(update);
  vis.createViewports(1,1);

  vis.setOutlineColor(0, 0, 0);

  vis.fitBox(0, 0, -dx*width/2, dx*width/2, -dx*height/2, dx*height/2);

  vis.setBackground(0, 0,
                    (GLfloat) 0, (GLfloat) 0, (GLfloat) 0);

  vis.pause(0, 0);

  vis.start();
}
#endif

double totalIters = 0;
double totalFlops = 0;
double totalBW    = 0;
double totalNS    = 0;

void run() {
  setupMesh();
  setupSolver();

  for (int i = 0; i < 50; i++)
    solve();

  std::ofstream avgFile("avg.dat");

  avgFile << totalFlops/totalIters << " " << totalBW/totalIters << " " << totalNS/totalIters << '\n';

  avgFile.close();
  exit(0);
}

void setupMesh() {
  occa::json settings = occa::json::loads("settings.json");

  deviceInfo = (std::string) settings["device"];

  stencilRadius = (int) settings["radius"];

  width  = (int) settings["width"];
  height = (int) settings["height"];

  Bx = (int) settings["bx"];
  By = (int) settings["by"];

  dx = (double) settings["dx"];
  dt = (double) settings["dt"];

  minU = (double) settings["minU"];
  maxU = (double) settings["maxU"];

  freq = (int) settings["frequency"];

  heightScale = 20000.0;

  currentTime = 0;

  std::cout << "Settings:\n" << settings << '\n';

  u1.resize(width*height, 0);
  u2.resize(width*height, 0);

  xyz.resize(2*width*height);

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      xyz[2*(h*width + w) + 0] = (w - width/2)*dx;
      xyz[2*(h*width + w) + 1] = (h - height/2)*dx;
    }
  }

  mX = width/2;
  mY = height/2;
}

#if OCCA_GL_ENABLED
void update() {
  if ( vis.keyIsPressed(' ') ) {
    if (!click) {
      solve();
      click = 1;
    }
  }
  else
    click = 0;

  vis.placeViewport(0,0);

  if (!vis.isPaused())
    solve();

  drawMesh();
}

const tFloat colorRange[18] = {1.0, 0.0, 0.0,
                              1.0, 1.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 1.0, 1.0,
                              0.0, 0.0, 1.0};

void getColor(tFloat *ret, tFloat scale, tFloat value, tFloat min) {
  tFloat c = (value - min)*scale;

  int b = ((int) 5.0*c) - ((int) c/1.0);

  tFloat ratio = 5.0*c - b;

  ret[0] = colorRange[3*b]   + ratio*(colorRange[3*(b+1)]   - colorRange[3*b]);
  ret[1] = colorRange[3*b+1] + ratio*(colorRange[3*(b+1)+1] - colorRange[3*b+1]);
  ret[2] = colorRange[3*b+2] + ratio*(colorRange[3*(b+1)+2] - colorRange[3*b+2]);
}

void getGrayscale(tFloat *ret, tFloat scale, tFloat value, tFloat min) {
  tFloat v = (maxU - value)*scale;

  ret[0] = v;
  ret[1] = v;
  ret[2] = v;
}

void drawMesh() {
  glBegin(GL_QUADS);

  tFloat color1[3];
  tFloat color2[3];
  tFloat color3[3];
  tFloat color4[3];

  tFloat ratio;

  if (minU == maxU)
    ratio = 0;
  else
    ratio = 1.0/(maxU - minU);

  for (int h = 0; h < (height-1); h++) {
    for (int w = 0; w < (width-1); w++) {
#if 0 // Matlab-like colors
      getColor(color1, ratio, u1[w     + (h)*width]    , minU);
      getColor(color2, ratio, u1[(w+1) + (h)*width]    , minU);
      getColor(color3, ratio, u1[(w+1) + ((h+1)*width)], minU);
      getColor(color4, ratio, u1[w     + ((h+1)*width)], minU);
#else // Grayscale
      getGrayscale(color1, ratio, u1[w     + (h)*width]    , minU);
      getGrayscale(color2, ratio, u1[(w+1) + (h)*width]    , minU);
      getGrayscale(color3, ratio, u1[(w+1) + ((h+1)*width)], minU);
      getGrayscale(color4, ratio, u1[w     + ((h+1)*width)], minU);
#endif

      glColor3f(color1[0], color1[1], color1[2]);
      glVertex3f(xyz[2*(w + (h)*width) + 0],
                 xyz[2*(w + (h)*width) + 1],
                 heightScale*u1[w + (h)*width]);

      glColor3f(color2[0], color2[1], color2[2]);
      glVertex3f(xyz[2*((w+1) + (h)*width) + 0],
                 xyz[2*((w+1) + (h)*width) + 1],
                 heightScale*u1[(w+1) + (h)*width]);

      glColor3f(color3[0], color3[1], color3[2]);
      glVertex3f(xyz[2*((w+1) + ((h+1)*width)) + 0],
                 xyz[2*((w+1) + ((h+1)*width)) + 1],
                 heightScale*u1[(w+1) + ((h+1)*width)]);

      glColor3f(color4[0], color4[1], color4[2]);
      glVertex3f(xyz[2*(w + ((h+1)*width)) + 0],
                 xyz[2*(w + ((h+1)*width)) + 1],
                 heightScale*u1[w + ((h+1)*width)]);
    }
  }

  glEnd();
}
#endif

/*
  Width : Number of nodes in the x direction
  Height: Number of nodes in the y direction

  dx: Spacing between nodes

  dt: Timestepping size

  u1, u2: Recordings

  mX/mY: Wavelet source point
 */

occa::device dev;
occa::memory o_u1, o_u2, o_u3;
occa::kernel fd2d;

void setupSolver() {
  dev.setup(deviceInfo);

  o_u1 = dev.malloc(u1.size()*sizeof(tFloat), &(u1[0]));
  o_u2 = dev.malloc(u1.size()*sizeof(tFloat), &(u1[0]));
  o_u3 = dev.malloc(u2.size()*sizeof(tFloat), &(u2[0]));

  occa::properties kernelProps;
  kernelProps["defines/sr"]   = stencilRadius;
  kernelProps["defines/w"]    = width;
  kernelProps["defines/h"]    = height;
  kernelProps["defines/dx"]   = dx;
  kernelProps["defines/dt"]   = dt;
  kernelProps["defines/freq"] = freq;
  kernelProps["defines/mX"]   = mX;
  kernelProps["defines/mY"]   = mY;
  kernelProps["defines/Bx"]   = Bx;
  kernelProps["defines/By"]   = By;

  if (sizeof(tFloat) == sizeof(float)) {
    kernelProps["defines/tFloat"] = "float";
  } else {
    kernelProps["defines/tFloat"] = "double";
  }

  fd2d = dev.buildKernel("fd2d.okl", "fd2d", kernelProps);
}

void solve() {
  const int iterations = 20;

  if (currentTime > 1) {
    dt = -dt;
  }
  if (currentTime < 0) {
    dt = -dt;
  }
  occa::streamTag startTag;
  occa::streamTag endTag;

  for (int i = 0; i < iterations; i++) {
    currentTime += dt;

    fd2d(o_u1, o_u2, o_u3, currentTime);
    o_u2.swap(o_u3);
    o_u1.swap(o_u2);
  }

  const int stencilDiameter = 2*stencilRadius + 1;

  const tFloat timeTakenPerIteration = 1.0;
  // const tFloat timeTakenPerIteration = fd2d.elapsed(); // Just measure 1 iteration
  const tFloat flops = width*height*(stencilDiameter*4 + 5);
  const tFloat bw    = sizeof(tFloat)*width*height*(2*stencilDiameter + 3);
  const tFloat ns    = width*height;

  totalFlops += flops/(1.0e9*timeTakenPerIteration);
  totalBW    += bw/(1.0e9*timeTakenPerIteration);
  totalNS    += ns/(1.0e6*timeTakenPerIteration);
  totalIters += 1;

  o_u1.copyTo( &(u1[0]) );
}
