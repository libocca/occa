#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include "math.h"

#include "visualizer.hpp"
#include "setupAide.hpp"

#include "occa.hpp"
#include "occa/array.hpp"

#if OCCA_GL_ENABLED
visualizer vis;
GLuint screenTexID;
#endif

setupAide setup;

const int    ITERATIONS     = 10;
const tFloat POWER          = 8;
const tFloat MAX_ITERATIONS = 5000;
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
tFloat pixel, halfPixel;

std::string deviceInfo;
occa::kernel rayMarcher;
occa::array<char> rgba;

void readSetupFile();
void setupRenderer();
void setupOCCA();
void setupGL();

void updateScene();

void render();

void updateRGB();
tFloat castShadow(const tFloat3 &rayLocation, const tFloat min, const tFloat max, const tFloat k);

std::string shapeFunctionName;
tFloat (*shapeFunction)(const tFloat3 &pos);
tFloat sphere(const tFloat3 &pos);
tFloat mandelbulb(const tFloat3 &pos);

#define PRINT_VEC(v)                                                    \
  std::cout << #v << " = (" << v.x << ", " << v.y << ", " << v.z << ")\n";

inline tFloat norm2(const tFloat3 &v) {
  return (v.x*v.x + v.y*v.y + v.z*v.z);
}

inline tFloat norm(const tFloat3 &v) {
  return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline tFloat3 normalize(const tFloat3 &v) {
  const tFloat vNorm = norm(v);
  if (vNorm < 0.00001f)
    return v;
  const tFloat invNorm = (1.0 / vNorm);
  return tFloat3(invNorm*v.x, invNorm*v.y, invNorm*v.z);
}

inline tFloat dot(const tFloat3 &a, const tFloat3 &b) {
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}

inline tFloat3 ortho(const tFloat3 &v) {
  const tFloat inv = 1.0 / sqrt(v.x*v.x + v.z*v.z);
  return tFloat3(-inv*v.z, v.y, inv*v.x);
}

inline tFloat3 cross(const tFloat3 &a, const tFloat3 &b) {
  return tFloat3(a.z*b.y - b.z*a.y,
                      a.x*b.z - b.x*a.z,
                      a.y*b.x - b.y*a.x);
}

inline tFloat clamp(const tFloat val, const tFloat min, const tFloat max) {
  return (val < min) ? min : ((max < val) ? max : val);
}

inline tFloat3 clamp(const tFloat3 &v, const tFloat min, const tFloat max) {
  return tFloat3(clamp(v.x, min, max),
                      clamp(v.y, min, max),
                      clamp(v.z, min, max));
}

inline tFloat3 saturate(const tFloat3 &v) {
  return clamp(v, 0.0f, 1.0f);
}

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
    updateRGB();
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
  setup.read("setuprc");

  setup.getArgs("DEVICE INFO", deviceInfo);

  setup.getArgs("SHAPE FUNCTION", shapeFunctionName);

  if(shapeFunctionName == "mandelbulb")
    shapeFunction = mandelbulb;
  else
    shapeFunction = sphere;

  setup.getArgs("WIDTH" , width);
  setup.getArgs("HEIGHT", height);

  std::cout << "DEVICE INFO    = " << deviceInfo        << '\n'
            << "SHAPE FUNCTION = " << shapeFunctionName << '\n'
            << "WIDTH          = " << width             << '\n'
            << "HEIGHT         = " << height            << '\n';
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

  const tFloat3 viewDirection = -normalize(nearFieldLocation);

  lightDirection = normalize(lightLocation);
  viewDirectionX = ortho(viewDirection);
  viewDirectionY = cross(viewDirectionX, viewDirection);
  eyeLocation    = nearFieldLocation - (EYE_DISTANCE_FROM_NEAR_FIELD * viewDirection); // NF - ReverseLocation

  pixel     = DEPTH_OF_FIELD / (0.5*(height + width));
  halfPixel = 0.5 * pixel;
}

void setupOCCA() {
  occa::setDevice(deviceInfo);

  rgba.allocate(4, width, height);

  for(int x = 0; x < width; ++x) {
    for(int y = 0; y < height; ++y) {
      rgba(3,x,y) = 255;
    }
  }

  occa::kernelInfo kInfo;

  kInfo.addDefine("WIDTH"         , width);
  kInfo.addDefine("HEIGHT"        , height);
  kInfo.addDefine("BATCH_SIZE"    , (width + 63)/ 64);
  kInfo.addDefine("SHAPE_FUNCTION", shapeFunctionName);
  kInfo.addDefine("PIXEL"         , pixel);
  kInfo.addDefine("HALF_PIXEL"    , halfPixel);

  if (sizeof(tFloat) == sizeof(float)) {
    kInfo.addDefine("tFloat" , "float");
    kInfo.addDefine("tFloat3", "float3");
  }
  else {
    kInfo.addDefine("tFloat" , "double");
    kInfo.addDefine("tFloat3", "double3");
  }

  rayMarcher = occa::buildKernel("rayMarcher.okl",
                                 "rayMarcher",
                                 kInfo);
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

  updateRGB();
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
  const double startTime = occa::currentTime();

  for (int y = 0; y < height; ++y) {
    const tFloat ny = y - 0.5*height;
    const tFloat3 p_viewDirectionY  = (ny*pixel) * viewDirectionY;

    for (int x = 0; x < width; ++x) {
      const tFloat nx = x - 0.5*width;
      const tFloat3 p_viewDirectionX = (nx*pixel) * viewDirectionX;

      tFloat3 rayLocation = (nearFieldLocation + p_viewDirectionY + p_viewDirectionX);
      const tFloat3 rayDirection = normalize(rayLocation - eyeLocation);

      tFloat d, distanceFromCamera;

      int iterations;
      for (iterations = 0; iterations < MAX_ITERATIONS; ++iterations) {
        d = shapeFunction(rayLocation);
        rayLocation += d * rayDirection;
        distanceFromCamera = norm(nearFieldLocation - rayLocation);

        if ((d < halfPixel) || (DEPTH_OF_FIELD <= distanceFromCamera))
          break;
      }

      if (distanceFromCamera < DEPTH_OF_FIELD) {
        const tFloat smallStep = 0.01f;
        const tFloat bigStep   = 0.02f;

        tFloat3 leftLoc, rightLoc;

        rayLocation.x -= smallStep;
        leftLoc.x = shapeFunction(rayLocation);
        rayLocation.x += bigStep;
        rightLoc.x = shapeFunction(rayLocation);
        rayLocation.x -= smallStep;

        rayLocation.y -= smallStep;
        leftLoc.y = shapeFunction(rayLocation);
        rayLocation.y += bigStep;
        rightLoc.y = shapeFunction(rayLocation);
        rayLocation.y -= smallStep;

        rayLocation.z -= smallStep;
        leftLoc.z = shapeFunction(rayLocation);
        rayLocation.z += bigStep;
        rightLoc.z = shapeFunction(rayLocation);
        rayLocation.z -= smallStep;

        const tFloat3 normal  = normalize(leftLoc - rightLoc);
        const tFloat3 halfway = normalize(rayDirection + lightDirection);
        const tFloat3 dotNH   = halfway * normal;

        const tFloat dotNL      = dot(lightDirection, normal);
        const tFloat diff       = clamp(dotNL, 0.0f, 1.0f);
        const tFloat spec       = pow(saturate(dotNH).x, 35.0f);
        const tFloat shad       = castShadow(rayLocation, 1.0f, DEPTH_OF_FIELD, 16.0f) + 0.25f;
        const tFloat brightness = (10.0f + (200.0f + spec*45.0f)*shad*diff) / 270.0f;

        rgba(0,x,y) = clamp(10 + (380 * brightness), 0, 255);
        rgba(1,x,y) = clamp(10 + (280 * brightness), 0, 255);
        rgba(2,x,y) = clamp(0  + (180 * brightness), 0, 255);
      }
      else {
        rgba(0,x,y) = 155 + clamp(1.5f*iterations, 0.0f, 100.0f);
        rgba(1,x,y) = 205 + clamp(1.5f*iterations, 0.0f,  50.0f);
        rgba(2,x,y) = 255;
      }
    }
  }

  const double endTime = occa::currentTime();

  std::cout << "Time Taken: " << (endTime - startTime) << '\n';
}

tFloat castShadow(const tFloat3 &rayLocation, const tFloat min, const tFloat max, const tFloat k) {
  tFloat ret = 1.0f;
  tFloat t = min;

  while (t < max) {
    const tFloat h = shapeFunction(rayLocation - (t * lightDirection));
    if (h < 0.001f)
      return 0.0f;

    const tFloat kht = (k * h / t);
    ret += (kht < ret)*(kht - ret);

    t += h;
  }

  return ret;
}

tFloat sphere(const tFloat3 &pos) {
  return norm(pos) - 1.5;
}

tFloat mandelbulb(const tFloat3 &pos) {
  tFloat3 z = pos;
  tFloat dr = 1.0;
  tFloat r  = 0.0;

  for (int i = 0; i < ITERATIONS; ++i) {
    r = norm(z);
    if (r > DEPTH_OF_FIELD)
      break;

    const tFloat theta    = POWER * acos(z.z / r);
    const tFloat phi      = POWER * atan2(z.y, z.x);
    const tFloat rPm1     = pow(r, POWER - 1);
    const tFloat zr       = r*rPm1;
    const tFloat sinTheta = sin(theta);
    dr = rPm1 * POWER * dr + 1.0;

    z.x = pos.x + zr*(sinTheta * cos(phi));
    z.y = pos.y + zr*(sinTheta * sin(phi));
    z.z = pos.z + zr*(cos(theta));
  }

  return (0.5 * r * log(r) / dr);
}
