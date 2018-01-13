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
#if OCCA_GL_ENABLED
#  ifndef OCCA_VISUALIZER_HEADER
#  define OCCA_VISUALIZER_HEADER

#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctime>

#include "occa.hpp"

#if   (OCCA_OS == OCCA_LINUX_OS)
#  include <GL/glew.h>
#  include <GL/gl.h>
#  include <GL/glut.h>
#elif (OCCA_OS == OCCA_OSX_OS)
#  include <OpenGL/gl.h>
#  include <GLUT/glut.h>
#elif (OCCA_OS == OCCA_WINDOWS_OS)
#  include <gl/glut.h>
#endif

using std::string;

class visualizer;
class viewport;

typedef void(visualizer::*eventFunction)(int, int);
typedef void(*functionPointer)();

#define OCCA_PI_H 1.570796326794896619231321
#define OCCA_PI   3.141592653589793238462643
#define OCCA_PI_2 6.283185307179586476925286

class viewport {
public:
  int width, height;
  int vpWOffset, vpHOffset;
  int vpWidth, vpHeight;

  GLfloat fov, aspect, znear, zfar;

  GLfloat eye[3], aim[3], up[3], right[2];
  GLfloat eyeOffset[3], oldEyeOffset[3];

  GLfloat camRadius, minCamRadius;
  GLfloat camTheta, camPhi;
  GLfloat oldCamTheta, oldCamPhi;

  int dragX, dragY, drag_f;

  GLfloat zoomSpeed;
  GLfloat panSpeed;
  GLfloat rotateSpeed;

  GLfloat bgColor[3];
  GLfloat lightPos[4];

  char light_f;
  char wireframe_f;

  viewport();
  ~viewport();

  void place();

  void drawBackground();

  void setAspectInformation(GLfloat, GLfloat, GLfloat);

  void resize(GLfloat, GLfloat, GLfloat, GLfloat);
  void resize(int, int, GLfloat, GLfloat, GLfloat, GLfloat);

  void placeCamera();

  void updateCamera();
  void phaseAngles();

  void setCamera(GLfloat, GLfloat, GLfloat);

  void setLight(char);

  void toggleLight();

  void toggleWireframeMode();

  GLint getPolyMode();

  void setBackground(GLfloat, GLfloat, GLfloat);
  void setBackground(int, int, int);

  void leftClick(int x, int y);
  void rightClick(int x, int y);

  void move(int x, int y);

  void updateSpeed();

  void moveForward();
  void moveBackward();
  void moveLeft();
  void moveRight();

  void rotateUp();
  void rotateDown();
  void rotateLeft();
  void rotateRight();

  void setRotateSpeed(GLfloat);
};

class visualizer {
public:

  //--[ SETTINGS ]--------------------------------
  string appName;

  eventFunction pressFunction[256];
  eventFunction releaseFunction[256];
  eventFunction mouseFunction[7][2];

  char keyState[256];
  char mouseState[7];

  GLfloat moveSpeed, moveSpeed_d;
  GLfloat rotateSpeed;

  char pause_f;
  char drag_f;
  char external_f;
  char lastKey_f;
  //==============================================


  //--[ DATA ]-----------------------------------
  GLfloat mx, my;
  //==============================================


  //--[ EXTERNAL DATA ]---------------------------
  float fps, fpsInv;

  functionPointer externalFunction;
  //==============================================


  //--[ COLORS ]----------------------------------
  static int colorCount;
  static GLfloat *palette;
  //==============================================


  //--[ CAMERA ]----------------------------------
  int rowVP, colVP;
  int vpRowCount, vpColCount;
  int *vpWidth;
  int *vpHeight;
  int oldVPWidth, oldVPHeight;

  int vpRowDrag, vpColDrag;
  int vpPadding;

  GLfloat vpOutlineColor[4];
  int vpOutlineSize;
  char vpOutline_f;

  viewport** viewports;

  GLfloat oldCamTheta, oldCamPhi;
  //==============================================


  //--[ VIEWPORTS ]----------------------------------
  int width, height;
  GLfloat fov, znear, zfar;
  int screenWidth, screenHeight;
  GLfloat viewX, viewY;
  //==============================================

  static visualizer *staticVisualizer;

  visualizer();
  visualizer(string);
  visualizer(string, int&, char**);
  ~visualizer();

  void setup(string);
  void setup(string, int&, char**);

  void init(int&, char**);
  void initFunctions();

  void start();

  void render();
  static void render_s() {
    staticVisualizer->render();
  }

  void resize(int, int);
  static void resize_s(int w, int h) {
    staticVisualizer->resize(w, h);
  };


  //--[ EXTERNAL DATA ]----------------------------------------------------
  void setFPS(float);

  void setExternalFunction(functionPointer);
  //=========================================================================


  //--[ EVENT FUNCTIONS ]----------------------------------------------------
  void keyPress(unsigned char, int, int);
  static void keyPress_s(unsigned char k, int x, int y) {
    staticVisualizer->keyPress(k, x, y);
  };

  void keyRelease(unsigned char, int, int);
  static void keyRelease_s(unsigned char k, int x, int y) {
    staticVisualizer->keyRelease(k, x, y);
  };

  void keyPressUpdate();

  void mousePress(int, int, int, int);
  static void mousePress_s(int button, int state, int x, int y) {
    staticVisualizer->mousePress(button, state, x, y);
  };

  void mouseDrag(int, int);
  static void mouseDrag_s(int x, int y) {
    staticVisualizer->mouseDrag(x, y);
  };

  void mouseMove(int, int);
  static void mouseMove_s(int x, int y) {
    staticVisualizer->mouseMove(x, y);
  };

  void setVPDrag(int*, int, int, int, int&, int&, int, int);
  int getVP();
  //=======================================================================


  //--[ CAMERA FUNCTIONS ]-------------------------------------------------
  void createViewports(int, int);
  void placeViewport(int, int);

  void fitBox(int, int, GLfloat, GLfloat, GLfloat, GLfloat);
  void fitBox(int, int, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat);

  void setCamera(int, int, GLfloat, GLfloat, GLfloat);

  void setLight(int, int, char);
  void toggleLight(int, int);

  void setBackground(int, int, GLfloat, GLfloat, GLfloat);
  void setBackground(int, int, int, int, int);
  void drawBackground(int, int);

  void placeCamera(int, int);
  void updateCamera(int, int);

  void drawBackground();

  void setOutlineColor(GLfloat, GLfloat, GLfloat);
  void setOutlineColor(GLfloat, GLfloat, GLfloat, GLfloat);
  void drawViewportOutlines();

  void startHUD();
  void endHUD();
  //=======================================================================


  //--[ MOUSE FUNCTIONS ]--------------------------------------------------
  void getRayNormal(int, int, GLfloat*, GLfloat*);

  void leftClickDown(int, int);
  void leftClickUp(int, int);

  void rightClickDown(int, int);
  void rightClickUp(int, int);
  //=======================================================================


  //--[ KEY FUNCTIONS ]----------------------------------------------------
  void voidEventFunction(int x, int y) {};
  void voidExternalFunction() {};

  char keyIsPressed(char);
  char lastKeyPressed();

  void toggleWireframeMode(int, int);

  void exitFunction(int, int);

  void pause(int, int);
  char isPaused();

  void moveForward(int, int);
  void moveBackward(int, int);
  void moveLeft(int, int);
  void moveRight(int, int);

  void rotateUp(int, int);
  void rotateDown(int, int);
  void rotateLeft(int, int);
  void rotateRight(int, int);

  void slowDown(int, int);
  void speedUp(int, int);
  //=======================================================================

  void setupColors();
  static void paletteColor(int, GLfloat);
};

#  endif
#endif
