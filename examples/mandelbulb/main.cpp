#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include "math.h"

#include "visualizer.hpp"

#include "occa.hpp"
#include "occa/array.hpp"

#if OCCA_GL_ENABLED
visualizer vis;
GLuint screenTexID;
#endif

const tFloat DEPTH_OF_FIELD = 2.5;
const tFloat EYE_DISTANCE_FROM_NEAR_FIELD = 2.2;
const tFloat DEG_TO_RAD = (M_PI / 180.0);

tFloat viewAngle  = 150.0 * DEG_TO_RAD;
tFloat lightAngle = 0;
tFloat3 lightDirection;
tFloat3 viewDirectionY, viewDirectionX;
tFloat3 nearFieldLocation;
tFloat3 eyeLocation;

int width, height;
int batchSize;
tFloat pixel, halfPixel;

std::string deviceInfo;
occa::kernel rayMarcher;
occa::array<char> rgba;

inline tFloat3 ortho(const tFloat3 &v) {
  const tFloat inv = 1.0 / sqrt(v.x*v.x + v.z*v.z);
  return tFloat3(-inv*v.z, v.y, inv*v.x);
}

void readSetupFile();
void setupRenderer();
void setupOCCA();
void setupGL();

void updateScene();

void render();

void updateRGB();
tFloat castShadow(const tFloat3 &rayLocation, const tFloat min, const tFloat max, const tFloat k);

std::string shapeFunction;

#if OCCA_GL_ENABLED
void glRun();
#endif

int main(int argc, char **argv) {
  readSetupFile();

  updateScene();
  setupOCCA();

#if OCCA_GL_ENABLED
  glRun();
#else
  while(true) {
    const double startTime = occa::currentTime();

    rayMarcher(rgba,
               lightDirection,
               viewDirectionY, viewDirectionX,
               nearFieldLocation, eyeLocation);

    occa::finish();

    const double endTime = occa::currentTime();

    std::cout << "Render Time Taken: " << (endTime - startTime) << '\n';

    updateScene();
  }
#endif

  return 0;
}

#if OCCA_GL_ENABLED
void glRun() {
  vis.setup("Raytracer Demo", width, height);

  vis.setExternalFunction(render);
  vis.createViewports(1,1);

  vis.getViewport(0,0).light_f     = false;
  vis.getViewport(0,0).hasCamera_f = false;

  vis.setOutlineColor(0, 0, 0);

  vis.setBackground(0, 0,
                    (GLfloat) 0, (GLfloat) 0, (GLfloat) 0);

  vis.pause(0, 0);

  setupRenderer();

  vis.start();
}
#endif

void readSetupFile() {
  occa::json settings = occa::json::loads("settings.json");

  deviceInfo = (std::string) settings["device"];

  shapeFunction = (std::string) settings["shapeFunction"];

  width  = (int) settings["width"];
  height = (int) settings["height"];

  batchSize = (int) settings["batchSize"];

  std::cout << settings << '\n';
}

void updateScene() {
  lightAngle    += DEG_TO_RAD;
  viewAngle     += DEG_TO_RAD;

  const tFloat3 lightLocation = tFloat3(0.5 * DEPTH_OF_FIELD * cos(lightAngle),
                                        0.5 * DEPTH_OF_FIELD,
                                        0.5 * DEPTH_OF_FIELD * sin(lightAngle));

  nearFieldLocation = tFloat3(0.5 * DEPTH_OF_FIELD * cos(viewAngle),
                              0,
                              0.5 * DEPTH_OF_FIELD * sin(viewAngle));

  const tFloat3 viewDirection = -occa::normalize(nearFieldLocation);

  lightDirection = occa::normalize(lightLocation);
  viewDirectionX = ortho(viewDirection);
  viewDirectionY = occa::cross(viewDirectionX, viewDirection);
  eyeLocation    = nearFieldLocation - (EYE_DISTANCE_FROM_NEAR_FIELD * viewDirection); // NF - ReverseLocation

  pixel     = DEPTH_OF_FIELD / (0.5*(height + width));
  halfPixel = 0.5 * pixel;
}

void setupOCCA() {
  occa::setDevice(deviceInfo);

  rgba.allocate(4, width, height);

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      rgba(3,x,y) = 255;
    }
  }

  occa::properties kernelProps;

  kernelProps["defines/WIDTH"]         = width;
  kernelProps["defines/HEIGHT"]        = height;
  kernelProps["defines/BATCH_SIZE"]    = batchSize;
  kernelProps["defines/SHAPE_FUNCTION"]= shapeFunction;
  kernelProps["defines/PIXEL"]         = pixel;
  kernelProps["defines/HALF_PIXEL"]    = halfPixel;

  if (sizeof(tFloat) == sizeof(float)) {
    kernelProps["defines/tFloat"]  = "float";
    kernelProps["defines/tFloat3"] = "float3";
  } else {
    kernelProps["defines/tFloat"]  = "double";
    kernelProps["defines/tFloat3"] = "double3";
  }

  rayMarcher = occa::buildKernel("rayMarcher.okl",
                                 "rayMarcher",
                                 kernelProps);
}

#if OCCA_GL_ENABLED
void setupRenderer() {
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &screenTexID);
  glBindTexture(GL_TEXTURE_2D, screenTexID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glDisable(GL_TEXTURE_2D);
}

void render() {
  vis.placeViewport(0,0);

  const double startTime = occa::sys::currentTime();

  rayMarcher(rgba,
             lightDirection,
             viewDirectionY, viewDirectionX,
             nearFieldLocation, eyeLocation);

  occa::finish();

  const double endTime = occa::sys::currentTime();

  std::cout << "Render Time Taken: " << (endTime - startTime) << '\n';

  updateScene();

  glColor3f(1.0, 1.0, 1.0);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, screenTexID);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

  glBegin(GL_QUADS);

  glTexCoord2f(0.0, 0.0); glVertex2f(-1,-1);
  glTexCoord2f(1.0, 0.0); glVertex2f( 1,-1);
  glTexCoord2f(1.0, 1.0); glVertex2f( 1, 1);
  glTexCoord2f(0.0, 1.0); glVertex2f(-1, 1);

  glEnd();
  glDisable(GL_TEXTURE_2D);
}
#endif

void updateRGB() {
}
