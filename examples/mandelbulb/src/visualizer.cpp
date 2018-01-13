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

#include "visualizer.hpp"

visualizer *visualizer::staticVisualizer;
int visualizer::colorCount;
GLfloat *visualizer::palette;

viewport::viewport() {
  light_f     = 0;
  wireframe_f = 0;
  hasCamera_f  = true;

  minCamRadius = 0.001;

  bgColor[0] = bgColor[1] = bgColor[2] = 1;

  lightPos[0] = lightPos[1] = 0;
  lightPos[2] = 4;
  lightPos[3] = 1;

  eyeOffset[0] = eyeOffset[1] = eyeOffset[2] = 0;

  setCamera(1, 1, 1);
}

viewport::~viewport() {}

void viewport::place() {
  glViewport(vpWOffset, (height - vpHeight) - vpHOffset, vpWidth, vpHeight);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  if (hasCamera_f)
    gluPerspective(fov, aspect, znear, zfar);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  if (hasCamera_f)
    placeCamera();

  drawBackground();

  if (wireframe_f) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }
  else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  if (light_f)
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
}

void viewport::drawBackground() {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, width, height, 0.0, -1.0, 10.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glDisable(GL_DEPTH_TEST);
  glBegin(GL_QUADS);

  glColor3f(bgColor[0], bgColor[1], bgColor[2]); glVertex2f(0, 0);
  glColor3f(bgColor[0], bgColor[1], bgColor[2]); glVertex2f(width, 0);
  glColor3f(bgColor[0], bgColor[1], bgColor[2]); glVertex2f(width, height);
  glColor3f(bgColor[0], bgColor[1], bgColor[2]); glVertex2f(0, height);

  glEnd();
  glEnable(GL_DEPTH_TEST);

  if (wireframe_f)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINES);
  else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void viewport::setAspectInformation(GLfloat _fov, GLfloat _znear, GLfloat _zfar) {
  fov   = _fov;
  znear = _znear;
  zfar  = _zfar;
}

void viewport::resize(GLfloat _vpWOffset, GLfloat _vpHOffset, GLfloat _vpWidth, GLfloat _vpHeight) {
  vpWOffset = _vpWOffset;
  vpHOffset = _vpHOffset;

  vpWidth  = _vpWidth;
  vpHeight = _vpHeight;

  aspect = ((GLfloat) _vpWidth)/((GLfloat) _vpHeight);
}

void viewport::resize(int _width, int _height, GLfloat _vpWOffset, GLfloat _vpHOffset, GLfloat _vpWidth, GLfloat _vpHeight) {
  width  = _width;
  height = _height;

  vpWOffset = _vpWOffset;
  vpHOffset = _vpHOffset;

  vpWidth  = _vpWidth;
  vpHeight = _vpHeight;

  aspect = ((GLfloat) _vpWidth)/((GLfloat) _vpHeight);
}

void viewport::placeCamera() {
  gluLookAt(eye[0], eye[1], eye[2],
            eye[0] + aim[0], eye[1] + aim[1], eye[2] + aim[2],
            up[0], up[1], up[2]);
}

void viewport::updateCamera() {
  phaseAngles();

  eye[0] = eyeOffset[0] + camRadius*cos(camTheta)*cos(camPhi);
  eye[1] = eyeOffset[1] + camRadius*sin(camTheta)*cos(camPhi);
  eye[2] = eyeOffset[2] + camRadius*sin(camPhi);

  aim[0] = -cos(camTheta)*cos(camPhi);
  aim[1] = -sin(camTheta)*cos(camPhi);
  aim[2] = -sin(camPhi);

  up[0] = cos(camTheta)*cos(camPhi + OCCA_PI_H);
  up[1] = sin(camTheta)*cos(camPhi + OCCA_PI_H);
  up[2] = sin(camPhi + OCCA_PI_H);

  right[0] = up[1]*aim[2] - up[2]*aim[1];
  right[1] = up[2]*aim[0] - up[0]*aim[2];

  GLfloat norm = 1.0/sqrt(right[0]*right[0] + right[1]*right[1]);

  right[0] *= norm;
  right[1] *= norm;

  updateSpeed();
}

void viewport::phaseAngles() {
  int cycles;

  cycles = camPhi/OCCA_PI_2;
  camPhi -= cycles*OCCA_PI_2;

  if (camPhi < 0)
    camPhi += OCCA_PI_2;

  cycles = camTheta/OCCA_PI_2;
  camTheta -= cycles*OCCA_PI_2;

  if (camTheta < 0)
    camTheta += OCCA_PI_2;
}

void viewport::setCamera(GLfloat x, GLfloat y, GLfloat z) {
  camRadius = sqrt(x*x + y*y + z*z);

  camPhi   = asin(z/camRadius);
  camTheta = atan2(y,x);

  updateCamera();
}

void viewport::setLight(char f) {
  light_f = f;

  if (light_f) {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
  }
  else
    glDisable(GL_LIGHTING);
}

void viewport::toggleLight() {
  light_f = !light_f;

  if (light_f) {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
  }
  else
    glDisable(GL_LIGHTING);
}

void viewport::toggleWireframeMode() {
  wireframe_f = !wireframe_f;
}

GLint viewport::getPolyMode() {
  GLint mode[2];

  // polyMode = (GLenum) mode;
  // mode[0] = GL_FRONT, GL_BACK or GL_FRONT_AND_BACK
  // mode[1] = GL_POINT, GL_LINE or GL_FILL
  glGetIntegerv(GL_POLYGON_MODE, mode);

  return mode[1];
}

void viewport::setBackground(GLfloat r, GLfloat g, GLfloat b) {
  bgColor[0] = r;
  bgColor[1] = g;
  bgColor[2] = b;
}

void viewport::setBackground(int r, int g, int b) {
  bgColor[0] = (float) r/255.0;
  bgColor[1] = (float) g/255.0;
  bgColor[2] = (float) b/255.0;
}

void viewport::leftClick(int x, int y) {
  dragX = x;
  dragY = y;

  oldCamPhi   = camPhi;
  oldCamTheta = camTheta;

  drag_f = 1;
}

void viewport::rightClick(int x, int y) {
  dragX = x;
  dragY = y;

  oldEyeOffset[0] = eyeOffset[0];
  oldEyeOffset[1] = eyeOffset[1];
  oldEyeOffset[2] = eyeOffset[2];

  drag_f = 0;
}

void viewport::move(int x, int y) {
  int dx = (x - dragX);
  int dy = (y - dragY);

  if (drag_f) {
    camPhi = oldCamPhi + dy*rotateSpeed;

    phaseAngles();

    int quad = oldCamPhi/OCCA_PI_H;

    if ( quad == 1 || quad == 2 )
      camTheta = oldCamTheta + dx*rotateSpeed;
    else
      camTheta = oldCamTheta - dx*rotateSpeed;

    updateCamera();
  }
  else {
    eyeOffset[0] = oldEyeOffset[0] + (dx*right[0] + dy*up[0])*panSpeed;
    eyeOffset[1] = oldEyeOffset[1] + (dx*right[1] + dy*up[1])*panSpeed;
    eyeOffset[2] = oldEyeOffset[2] + dy*up[2]*panSpeed;

    eye[0] = eyeOffset[0] + camRadius*cos(camTheta)*cos(camPhi);
    eye[1] = eyeOffset[1] + camRadius*sin(camTheta)*cos(camPhi);
    eye[2] = eyeOffset[2] + camRadius*sin(camPhi);
  }
}

void viewport::updateSpeed() {
  GLfloat speedFactor = 1.0/200.0;

  panSpeed  = camRadius*tan(fov)/vpWidth;
  zoomSpeed = speedFactor*camRadius;
}

void viewport::moveForward() {
  camRadius -= zoomSpeed;

  if (camRadius < minCamRadius)
    camRadius = minCamRadius;

  updateCamera();
  updateSpeed();
}

void viewport::moveBackward() {
  camRadius += zoomSpeed;

  updateCamera();
  updateSpeed();
}

void viewport::moveLeft() {
  eyeOffset[0] -= zoomSpeed*right[0];
  eyeOffset[1] -= zoomSpeed*right[1];

  updateCamera();
}

void viewport::moveRight() {
  eyeOffset[0] += zoomSpeed*right[0];
  eyeOffset[1] += zoomSpeed*right[1];

  updateCamera();
}

void viewport::rotateUp() {
  camPhi += rotateSpeed;
  updateCamera();
}

void viewport::rotateDown() {
  camPhi -= rotateSpeed;
  updateCamera();
}

void viewport::rotateLeft() {
  camTheta += rotateSpeed;
  updateCamera();
}

void viewport::rotateRight() {
  camTheta -= rotateSpeed;
  updateCamera();
}

void viewport::setRotateSpeed(GLfloat rs) {
  rotateSpeed = rs;
}

visualizer::visualizer() {}

visualizer::visualizer(string name, int inputWidth, int inputHeight) {
  setup(name, inputWidth, inputHeight);
}

visualizer::visualizer(string name, int inputWidth, int inputHeight, int& argc, char **argv) {
  setup(name, inputWidth, inputHeight, argc, argv);
}

visualizer::~visualizer() {}

void visualizer::setup(string name, int inputWidth, int inputHeight) {
  int argc     = 1;
  char arg[]   = "fake";
  char *argv[] = { arg , NULL };

  setup(name, inputWidth, inputHeight, argc, argv);
}

void visualizer::setup(string name, int inputWidth, int inputHeight, int& argc, char **argv) {
  staticVisualizer = this;

  //--[ SETTINGS ]--------------------------------
  appName = name;

  for (int i=0; i<256; i++) {
    pressFunction[i]   = &visualizer::voidEventFunction;
    releaseFunction[i] = &visualizer::voidEventFunction;

    keyState[i] = 0;
  }

  for (int i=0; i<7; i++) {
    mouseFunction[i][0] = &visualizer::voidEventFunction;
    mouseFunction[i][1] = &visualizer::voidEventFunction;

    mouseState[i] = 0;
  }

  rotateSpeed = OCCA_PI/600.0;

  pause_f     = 0;
  external_f  = 0;
  drag_f      = 0;
  //==============================================


  //--[ COLORS ]----------------------------------
  setupColors();
  //==============================================


  //--[ CAMERA ]----------------------------------
  rowVP = colVP = 0;

  vpRowCount = vpColCount = 0;

  vpRowDrag = vpColDrag = 0;

  vpPadding = 3;

  vpOutlineColor[0] = 0;
  vpOutlineColor[1] = 0;
  vpOutlineColor[2] = 0;
  vpOutlineColor[3] = 0;

  vpOutlineSize = 1;
  vpOutline_f   = 1;
  //==============================================


  //--[ VIEWPORTS ]----------------------------------
  width  = inputWidth;
  height = inputHeight;

  fov    = 45.0;
  znear  = 0.0001;
  zfar   = 10000;
  //==============================================

  init(argc, argv);
  initFunctions();
}

void visualizer::init(int& argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

  screenWidth  = glutGet(GLUT_SCREEN_WIDTH);
  screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

  glutInitWindowPosition( (screenWidth - width)/2 , (screenHeight - height)/2 );
  glutInitWindowSize(width, height);
  glutCreateWindow(appName.c_str());

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glEnable(GL_DEPTH_TEST);

  glEnable(GL_COLOR_MATERIAL);

  glShadeModel(GL_SMOOTH);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glutDisplayFunc(render_s);
  glutIdleFunc(render_s);
  glutReshapeFunc(resize_s);
  glutKeyboardFunc(keyPress_s);
  glutKeyboardUpFunc(keyRelease_s);
  glutMouseFunc(mousePress_s);
  glutMotionFunc(mouseDrag_s);
  glutPassiveMotionFunc(mouseMove_s);
}

void visualizer::initFunctions() {
  pressFunction['q'] = &visualizer::exitFunction;
  pressFunction[27]  = &visualizer::exitFunction;

  pressFunction['p'] = &visualizer::pause;

  pressFunction['-'] = &visualizer::moveBackward;
  pressFunction['_'] = &visualizer::moveBackward;
  pressFunction['='] = &visualizer::moveForward;
  pressFunction['+'] = &visualizer::moveForward;

  pressFunction['m'] = &visualizer::toggleWireframeMode;

  mouseFunction[0][0] = &visualizer::leftClickDown;
  mouseFunction[0][1] = &visualizer::leftClickUp;

  mouseFunction[2][0] = &visualizer::rightClickDown;
  mouseFunction[2][1] = &visualizer::rightClickUp;
}

void visualizer::start() {
  if (!vpRowCount || !vpColCount) {
    printf("Viewports not setup [ createViewports(int,int) ] .\nExiting Program.\n");
    throw 1;
  }

  glutMainLoop();
}

void visualizer::render() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  keyPressUpdate();

  if (external_f)
    externalFunction();

  if (vpOutline_f)
    drawViewportOutlines();

  glFlush();

  glutSwapBuffers();
}

void visualizer::resize(int w, int h) {
  for (int v=0; v<vpRowCount; v++)
    vpHeight[v+1] = vpHeight[v] + ( (float) (vpHeight[v+1] - vpHeight[v])*((float) h) )/((float) height);

  for (int v=0; v<vpColCount; v++)
    vpWidth[v+1] = vpWidth[v]  + ( (float) (vpWidth[v+1]  - vpWidth[v])*((float) w) )/((float) width);

  for (int r=0; r<vpRowCount; r++)
    for (int c=0; c<vpColCount; c++)
      viewports[r][c].resize(w, h, vpWidth[c], vpHeight[r],
                             vpWidth[c+1]  - vpWidth[c],
                             vpHeight[r+1] - vpHeight[r]);

  width  = w;
  height = h;

  glutPostRedisplay();
}


//--[ EXTERNAL DATA ]----------------------------------------------------
void visualizer::setFPS(float f) {
  fps    = f;
  fpsInv = 1.0/f;
}

void visualizer::setExternalFunction(functionPointer fp) {
  external_f = 1;

  externalFunction = fp;
}
//=========================================================================


//--[ EVENT FUNCTIONS ]----------------------------------------------------
void visualizer::keyPress(unsigned char k, int x, int y) {
  lastKey_f = k;

  staticVisualizer->keyState[k] = 1;
  (staticVisualizer->*pressFunction[(int) k])(x, y);
}

void visualizer::keyRelease(unsigned char k, int x, int y) {
  staticVisualizer->keyState[k] = 0;
  (staticVisualizer->*releaseFunction[(int) k])(x, y);
}

void visualizer::keyPressUpdate() {
  if (keyState['-']) moveBackward(0,0);
  if (keyState['_']) moveBackward(0,0);
  if (keyState['=']) moveForward(0,0);
  if (keyState['+']) moveForward(0,0);
}

void visualizer::mousePress(int button, int state, int x, int y) {
  staticVisualizer->mouseState[button] = !state;
  (staticVisualizer->*mouseFunction[button][state])(x, y);
}

void visualizer::mouseDrag(int x, int y) {
  if (mouseState[0]) {
    drag_f = 1;

    if (vpRowDrag) {
      int currentRow = vpRowDrag - 1;

      vpHeight[currentRow] = oldVPHeight + (y - my);

      int vpMax = vpHeight[currentRow+1] - 2*vpPadding;
      int vpMin = vpHeight[currentRow-1]   + 2*vpPadding;

      if ( vpHeight[currentRow] >= vpMax )
        vpHeight[currentRow] = vpMax;
      else if ( vpHeight[currentRow] <= vpMin )
        vpHeight[currentRow] = vpMin;

      for (int c=0; c<vpColCount; c++) {
        viewports[currentRow-1][c].resize(vpWidth[c], vpHeight[currentRow-1],
                                          vpWidth[c+1] - vpWidth[c],
                                          vpHeight[currentRow] - vpHeight[currentRow-1]);

        viewports[currentRow][c].resize(vpWidth[c], vpHeight[currentRow],
                                        vpWidth[c+1] - vpWidth[c],
                                        vpHeight[currentRow+1] - vpHeight[currentRow]);
      }
    }
    if (vpColDrag) {
      int currentCol = vpColDrag-1;

      vpWidth[currentCol] = oldVPWidth + (x - mx);

      int vpMax = vpWidth[currentCol+1] - 2*vpPadding;
      int vpMin = vpWidth[currentCol-1]   + 2*vpPadding;

      if ( vpWidth[currentCol] >= vpMax )
        vpWidth[currentCol] = vpMax;
      else if ( vpWidth[currentCol] <= vpMin )
        vpWidth[currentCol] = vpMin;

      for (int r=0; r<vpRowCount; r++) {
        viewports[r][currentCol-1].resize(vpWidth[currentCol-1], vpHeight[r],
                                          vpWidth[currentCol] - vpWidth[currentCol-1],
                                          vpHeight[r+1] - vpHeight[r]);

        viewports[r][currentCol].resize(vpWidth[currentCol], vpHeight[r],
                                        vpWidth[currentCol+1] - vpWidth[currentCol],
                                        vpHeight[r+1] - vpHeight[r]);
      }
    }
    if (!(vpRowDrag || vpColDrag))
      viewports[rowVP][colVP].move(x, y);
  }
  if (mouseState[2])
    viewports[rowVP][colVP].move(x, y);
}

void visualizer::mouseMove(int x, int y) {
  mx = x;
  my = y;

  for (int v=0; v<vpRowCount; v++) {
    if ( y <= vpHeight[v+1]) {
      rowVP = v;
      break;
    }
  }

  for (int v=0; v<vpColCount; v++) {
    if (x <= vpWidth[v+1]) {
      colVP = v;
      break;
    }
  }

  setVPDrag(vpWidth , x, colVP, vpColCount, oldVPWidth , vpColDrag, GLUT_CURSOR_INHERIT, GLUT_CURSOR_LEFT_RIGHT);

  if (!vpColDrag)
    setVPDrag(vpHeight, y, rowVP, vpRowCount, oldVPHeight, vpRowDrag, GLUT_CURSOR_INHERIT, GLUT_CURSOR_UP_DOWN);
  else
    setVPDrag(vpHeight, y, rowVP, vpRowCount, oldVPHeight, vpRowDrag, GLUT_CURSOR_LEFT_RIGHT, GLUT_CURSOR_CROSSHAIR);
}

void visualizer::setVPDrag(int* vpOffset, int xy, int currentVP, int vpCount, int &oldOffset, int &drag, int defaultCursor, int cursor) {
  if ( (vpOffset[currentVP] - vpPadding) <= xy && xy <= (vpOffset[currentVP] + vpPadding) ) {
    glutSetCursor(cursor);
    oldOffset = vpOffset[currentVP];

    // Don't drag the left/top side of the window
    drag = ( currentVP % vpCount ) ? (currentVP + 1) : 0;
  }
  else if ( (vpOffset[currentVP+1] - vpPadding) <= xy && xy <= (vpOffset[currentVP+1] + vpPadding) ) {
    glutSetCursor(cursor);
    oldOffset = vpOffset[currentVP+1];

    // Don't drag the right/bottom side of the window
    drag = (currentVP < (vpCount-1)) ? (currentVP + 2) : 0;
  }
  else {
    glutSetCursor(defaultCursor);
    drag = 0;
  }
}

int visualizer::getVP() {
  return ( vpColCount*rowVP + colVP );
}
//=========================================================================


//--[ CAMERA FUNCTIONS ]---------------------------------------------------
void visualizer::createViewports(int rCount, int cCount) {
  if (rCount <= 0 || cCount <= 0) {
    printf("Number of viewports [ %d , %d ] is not acceptable.\nExiting Program.\n", rCount, cCount);
    throw 1;
  }

  vpRowCount = rCount;
  vpColCount = cCount;

  int dw = ((float) width)/((float) cCount);
  int dh = ((float) height)/((float) rCount);

  vpWidth  = new int[rCount+1];
  vpHeight = new int[cCount+1];

  for (int v=0; v<=cCount; v++)
    vpWidth[v]  = v*dw;

  for (int v=0; v<=rCount; v++)
    vpHeight[v] = v*dh;

  viewports = new viewport*[vpRowCount];
  for (int v=0; v<vpRowCount; v++)
    viewports[v] = new viewport[vpColCount];

  for (int r=0; r<rCount; r++) {
    for (int c=0; c<cCount; c++) {
      viewports[r][c].setAspectInformation(fov, znear, zfar);
      viewports[r][c].resize(width, height,
                             vpWidth[c], vpHeight[r],
                             vpWidth[c+1]  - vpWidth[c],
                             vpHeight[r+1] - vpHeight[r]);

      viewports[r][c].setRotateSpeed(rotateSpeed);

      viewports[r][c].setCamera(1,1,1);

      viewports[r][c].setLight(0);

      viewports[r][c].setBackground(255, 255, 255);
    }
  }
}


void visualizer::fitBox(int r, int c, GLfloat x1, GLfloat x2, GLfloat y1, GLfloat y2) {
  GLfloat fov    = viewports[r][c].fov;
  GLfloat aspect = viewports[r][c].aspect;

  GLfloat z1 = (y2 - y1)/(2*tan(fov*OCCA_PI/360.0));
  GLfloat z2 = (x2 - x1)/(2*tan(fov*aspect*OCCA_PI/360.0));

  GLfloat x = (x1 + x2)/2.0;
  GLfloat y = (y1 + y2)/2.0;
  GLfloat z = z1 > z2 ? z1 : z2;

  viewports[r][c].setCamera(x,y,z);
}

void visualizer::fitBox(int r, int c, GLfloat x1, GLfloat x2, GLfloat y1, GLfloat y2, GLfloat z1, GLfloat z2) {
  GLfloat* up    = viewports[r][c].up;
  GLfloat* right = viewports[r][c].right;
}

void visualizer::placeViewport(int r, int c) {
  viewports[r][c].place();
}

void visualizer::setCamera(int r, int c, GLfloat x, GLfloat y, GLfloat z) {
  viewports[r][c].setCamera(x,y,z);
}

void visualizer::setLight(int r, int c, char f) {
  viewports[r][c].setLight(f);
}

void visualizer::toggleLight(int r, int c) {
  viewports[r][c].toggleLight();
}

void visualizer::setBackground(int r, int c, GLfloat red, GLfloat green, GLfloat blue) {
  viewports[r][c].setBackground(red, green, blue);
}

void visualizer::setBackground(int r, int c, int red, int green, int blue) {
  viewports[r][c].setBackground(red, green, blue);
}

void visualizer::drawBackground(int r, int c) {
  viewports[r][c].drawBackground();
}

void visualizer::placeCamera(int r, int c) {
  viewports[r][c].placeCamera();
}

void visualizer::updateCamera(int r, int c) {
  viewports[r][c].updateCamera();
}

void visualizer::drawBackground() {
  glViewport(0, 0, width, height);

  gluLookAt(0, 0, 0,
            0, 0, 1,
            0, 1, 0);

  startHUD();

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glBegin(GL_QUADS);

  glColor3f(0, 0, 0); glVertex2f(0, 0);
  glColor3f(0, 0, 0); glVertex2f(width, 0);
  glColor3f(0, 0, 0); glVertex2f(width, height);
  glColor3f(0, 0, 0); glVertex2f(0, height);

  glEnd();

  endHUD();
}

void visualizer::setOutlineColor(GLfloat r, GLfloat g, GLfloat b) {
  setOutlineColor(r,g,b,1);
}

void visualizer::setOutlineColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
  vpOutlineColor[0] = r;
  vpOutlineColor[1] = g;
  vpOutlineColor[2] = b;
  vpOutlineColor[3] = a;
}

void visualizer::drawViewportOutlines() {
  glViewport(0, 0, width, height);

  gluLookAt(0, 0, 0,
            0, 0, 1,
            0, 1, 0);

  GLfloat* color = vpOutlineColor;

  startHUD();

  glBegin(GL_QUADS);

  for (int r=1; r<vpRowCount; r++) {
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(0    , vpHeight[r] - vpOutlineSize);
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(0    , vpHeight[r] + vpOutlineSize);
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(width, vpHeight[r] + vpOutlineSize);
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(width, vpHeight[r] - vpOutlineSize);
  }

  for (int c=1; c<vpColCount; c++) {
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(vpWidth[c] - vpOutlineSize, 0);
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(vpWidth[c] + vpOutlineSize, 0);
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(vpWidth[c] + vpOutlineSize, height);
    glColor4f(color[0], color[1], color[2], color[3]); glVertex2f(vpWidth[c] - vpOutlineSize, height);
  }

  glEnd();

  endHUD();
}

void visualizer::startHUD() {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, width, height, 0.0, -1.0, 10.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_DEPTH_TEST);
}

void visualizer::endHUD() {
  glEnable(GL_DEPTH_TEST);

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}
//=========================================================================


//--[ MOUSE FUNCTIONS ]----------------------------------------------------
void visualizer::getRayNormal(int x, int y, GLfloat *bgPos, GLfloat *normal) {
  GLint viewport[4];
  GLdouble modelview[16], projection[16];
  GLdouble winX, winY;
  GLdouble bg2[3], fg2[3];
  GLfloat norm = 0.0;

  glGetDoublev( GL_MODELVIEW_MATRIX , modelview );
  glGetDoublev( GL_PROJECTION_MATRIX, projection );
  glGetIntegerv(GL_VIEWPORT         , viewport );

  winX = x;
  winY = viewport[3] - y;

  gluUnProject( winX, winY, 0, modelview, projection, viewport, bg2, bg2+1, bg2+2);
  gluUnProject( winX, winY, 1, modelview, projection, viewport, fg2, fg2+1, fg2+2);

  for (int i=0; i<16; i++)
    modelview[i] = 0.0;
  for (int i=0; i<4; i++)
    modelview[4*i+3] = 1.0;
  for (int i=0; i<3; i++) {
    modelview[i]   = bg2[i];
    modelview[4+i] = fg2[i];
  }

  glPushMatrix();

  glMultMatrixd(modelview);
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);

  glPopMatrix();

  for (int i=0; i<3; i++) {
    bgPos[i]  = modelview[i];
    normal[i] = modelview[4+i] - modelview[i];
    norm += normal[i]*normal[i];
  }

  norm = 1.0/sqrt(norm);

  for (int i=0; i<3; i++)
    normal[i] *= norm;
}

void visualizer::leftClickDown(int x, int y) {
  mx = x;
  my = y;

  viewports[rowVP][colVP].leftClick(x,y);
}

void visualizer::leftClickUp(int x, int y) {
  drag_f = 0;
}

void visualizer::rightClickDown(int x, int y) {
  mx = x;
  my = y;

  viewports[rowVP][colVP].rightClick(x,y);
}

void visualizer::rightClickUp(int x, int y) {
  drag_f = 0;
}
//=========================================================================


//--[ KEY FUNCTIONS ]----------------------------------------------------
char visualizer::keyIsPressed(char c) {
  return keyState[c];
}

char visualizer::lastKeyPressed() {
  return lastKey_f;
}

void visualizer::toggleWireframeMode(int x, int y) {
  viewports[rowVP][colVP].toggleWireframeMode();
}

void visualizer::exitFunction(int x, int y) {
  exit(1);
}

void visualizer::pause(int x, int y) {
  pause_f = !pause_f;
}

char visualizer::isPaused() {
  return pause_f;
}

void visualizer::moveForward(int x, int y) {
  viewports[rowVP][colVP].moveForward();
}

void visualizer::moveBackward(int x, int y) {
  viewports[rowVP][colVP].moveBackward();
}

void visualizer::moveLeft(int x, int y) {
  viewports[rowVP][colVP].moveLeft();
}

void visualizer::moveRight(int x, int y) {
  viewports[rowVP][colVP].moveRight();
}

void visualizer::rotateUp(int x, int y) {
  viewports[rowVP][colVP].rotateUp();
}

void visualizer::rotateDown(int x, int y) {
  viewports[rowVP][colVP].rotateDown();
}

void visualizer::rotateLeft(int x, int y) {
  viewports[rowVP][colVP].rotateLeft();
}

void visualizer::rotateRight(int x, int y) {
  viewports[rowVP][colVP].rotateRight();
}

void visualizer::setupColors() {
  colorCount = 14;
  palette = new GLfloat[3*colorCount];

  palette[3*0  + 0] = ( 72/255.0); palette[3*0  + 1] = (118/255.0); palette[3*0  + 2] = (255/255.0);
  palette[3*1  + 0] = (  0/255.0); palette[3*1  + 1] = (238/255.0); palette[3*1  + 2] = (118/255.0);
  palette[3*2  + 0] = (238/255.0); palette[3*2  + 1] = (238/255.0); palette[3*2  + 2] = (  0/255.0);
  palette[3*3  + 0] = (238/255.0); palette[3*3  + 1] = (201/255.0); palette[3*3  + 2] = (  0/255.0);
  palette[3*4  + 0] = (255/255.0); palette[3*4  + 1] = (193/255.0); palette[3*4  + 2] = (193/255.0);
  palette[3*5  + 0] = (142/255.0); palette[3*5  + 1] = ( 58/255.0); palette[3*5  + 2] = (142/255.0);
  palette[3*6  + 0] = (192/255.0); palette[3*6  + 1] = (192/255.0); palette[3*6  + 2] = (192/255.0);
  palette[3*7  + 0] = (219/255.0); palette[3*7  + 1] = (112/255.0); palette[3*7  + 2] = (147/255.0);
  palette[3*8  + 0] = (255/255.0); palette[3*8  + 1] = (187/255.0); palette[3*8  + 2] = (255/255.0);
  palette[3*9  + 0] = ( 61/255.0); palette[3*9  + 1] = ( 89/255.0); palette[3*9  + 2] = (171/255.0);
  palette[3*10 + 0] = (100/255.0); palette[3*10 + 1] = (149/255.0); palette[3*10 + 2] = (237/255.0);
  palette[3*11 + 0] = (  0/255.0); palette[3*11 + 1] = (134/255.0); palette[3*11 + 2] = (139/255.0);
  palette[3*12 + 0] = (238/255.0); palette[3*12 + 1] = (216/255.0); palette[3*12 + 2] = (174/255.0);
  palette[3*13 + 0] = (244/255.0); palette[3*13 + 1] = (164/255.0); palette[3*13 + 2] = ( 96/255.0);
}

void visualizer::paletteColor(int c, GLfloat alpha) {
  int off = 3*(c % colorCount);

  glColor4f(palette[off], palette[off+1], palette[off+2], alpha);
}
//=========================================================================

#endif
