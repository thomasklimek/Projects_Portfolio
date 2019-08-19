#include "Camera.h"
#include <iostream>

using namespace std;

Camera::Camera() {
	reset();
}

Camera::~Camera() {
}

void Camera::reset() {
	orientLookAt(glm::vec3(0.0f, 0.0f, DEFAULT_FOCUS_LENGTH), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	setViewAngle(VIEW_ANGLE);
	setNearPlane(NEAR_PLANE);
	setFarPlane(FAR_PLANE);
	//screenWidth = screenHeight = 200;
	screenWidthRatio = 1.0f;
	rotU = rotV = rotW = 0;
	//eyeX = eyeY = eyeZ = 0.0f;
}

//called by main.cpp as a part of the slider callback for controlling rotation
// the reason for computing the diff is to make sure that we are only incrementally rotating the camera
void Camera::setRotUVW(float u, float v, float w) {
	float diffU = u - rotU;
	float diffV = v - rotV;
	float diffW = w - rotW;
	rotateU(diffU);
	rotateV(diffV);
	rotateW(diffW);
	rotU = u;
	rotV = v;
	rotW = w;
}


void Camera::orientLookAt(glm::vec3 eyePoint, glm::vec3 lookatPoint, glm::vec3 _upVec) {
	glm::vec3 _lookVec = glm::normalize(glm::vec3(lookatPoint.x - eyePoint.x, lookatPoint.y - eyePoint.y, lookatPoint.z - eyePoint.z));
	orientLookVec(eyePoint, _lookVec, _upVec);
}

void Camera::orientLookVec(glm::vec3 eyePoint, glm::vec3 _lookVec, glm::vec3 _upVec) {

	lookVec = _lookVec;
	upVec = _upVec;

	eyeX = eyePoint.x;
	eyeY = eyePoint.y;
	eyeZ = eyePoint.z;

	// calculate u, v, w from lookVec and eyePoint
	wVec = glm::normalize(-_lookVec);
	uVec = glm::normalize(glm::cross(_upVec, wVec));
	vVec = glm::cross(wVec, uVec);
}

glm::mat4 Camera::getScaleMatrix() {
	glm::mat4 M2(1.0f);

	//M2[0][0] = 1.0 / (tan(viewAngle * PI / 360.0)*farPlane*screenWidthRatio);
	//M2[1][1] = 1.0 / (tan(viewAngle * PI / 360.0)*farPlane);
	M2[0][0] = 1 / (tan(glm::radians(viewAngle) / 2) * farPlane);

	// calculate height angle based off of aspect ratio
	M2[1][1] = screenWidthRatio / (tan(glm::radians(viewAngle) / 2) * farPlane);



	M2[2][2] = 1 / farPlane;
	
	return M2;
}

glm::mat4 Camera::getInverseScaleMatrix() {
	return glm::inverse(getScaleMatrix());
}

glm::mat4 Camera::getUnhingeMatrix() {
	glm::mat4 M1(1.0f);

	float c = -nearPlane / farPlane;

	M1[2][2] = -1.0f / (c + 1.0f);
	M1[3][2] = c / (c + 1.0f);
	M1[2][3] = -1.0f;
	M1[3][3] = 0.0f;
	
	return M1;
}


glm::mat4 Camera::getProjectionMatrix() {
	glm::mat4 projMat4(1.0);
	return getUnhingeMatrix() * getScaleMatrix();
}

glm::mat4 Camera::getInverseModelViewMatrix() {
	glm::mat4 invModelViewMat4(1.0);
	return glm::inverse(getModelViewMatrix());
}


void Camera::setViewAngle (float _viewAngle) {
	viewAngle = _viewAngle + 6;
}

void Camera::setNearPlane (float _nearPlane) {
	nearPlane = _nearPlane;
}

void Camera::setFarPlane (float _farPlane) {
	farPlane = _farPlane;
}

void Camera::setScreenSize (int _screenWidth, int _screenHeight) {
	screenWidth = _screenWidth;
	screenHeight = _screenHeight;

	screenWidthRatio = (float)screenWidth / (float)screenHeight;
}

glm::mat4 Camera::getModelViewMatrix() {
	glm::mat4 modelViewMat4(1.0);

	// eyePoint translation matrix
	glm::mat4 M4 = glm::translate(glm::mat4(1.0f), glm::vec3(-eyeX, -eyeY, -eyeZ));

	// Rxyz2uvw matrix
	glm::mat4 M3(1.0f);
	M3[0][0] = uVec.x;
	M3[1][0] = uVec.y;
	M3[2][0] = uVec.z;
	M3[0][1] = vVec.x;
	M3[1][1] = vVec.y;
	M3[2][1] = vVec.z;
	M3[0][2] = wVec.x;
	M3[1][2] = wVec.y;
	M3[2][2] = wVec.z;

	return M3 * M4;
}


void Camera::rotateV(float degrees) {
	rotate(glm::vec3(eyeX, eyeY, eyeZ), vVec, degrees);
}

void Camera::rotateU(float degrees) {
	rotate(glm::vec3(eyeX, eyeY, eyeZ), uVec, degrees);
}

void Camera::rotateW(float degrees) {
	rotate(glm::vec3(eyeX, eyeY, eyeZ), wVec, -1.0f * degrees);
}

void Camera::translate(glm::vec3 v) {
	eyeX = eyeX + v.x;
	eyeY = eyeY + v.y;
	eyeZ = eyeZ + v.z;
}

void Camera::rotate(glm::vec3 point, glm::vec3 axis, float degrees) {
	// translate camera by -point
	// rotate eyePoint by degrees around axis
	// rotate u, v, w, upVec, lookVec by degrees around axis
	// translate back

	glm::vec3 eyeVec = glm::vec3(eyeX - point.x, eyeY - point.y, eyeZ - point.z);

	glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(degrees), axis);

	eyeVec = glm::vec3(rotationMatrix * glm::vec4(eyeVec, 1.0f));

	uVec = glm::vec3(rotationMatrix * glm::vec4(uVec, 1.0f));
	vVec = glm::vec3(rotationMatrix * glm::vec4(vVec, 1.0f));
	wVec = glm::vec3(rotationMatrix * glm::vec4(wVec, 1.0f));
	lookVec = glm::vec3(rotationMatrix * glm::vec4(lookVec, 1.0f));
	upVec = glm::vec3(rotationMatrix * glm::vec4(upVec, 1.0f));

	eyeX = eyeVec.x + point.x;
	eyeY = eyeVec.y + point.y;
	eyeZ = eyeVec.z + point.z;
}

glm::vec3 Camera::getEyePoint() {
	glm::vec3 eyeVec3 = glm::vec3(eyeX, eyeY, eyeZ); 
	return eyeVec3;
}

glm::vec3 Camera::getLookVector() {
	return lookVec;
}

glm::vec3 Camera::getUpVector() {
	return upVec;
}

float Camera::getViewAngle() {
	return viewAngle;
}

float Camera::getNearPlane() {
	return nearPlane;
}

float Camera::getFarPlane() {
	return farPlane;
}

int Camera::getScreenWidth() {
	return screenWidth;
}

int Camera::getScreenHeight() {
	return screenHeight;
}

float Camera::getScreenWidthRatio() {
	return screenWidthRatio;
}
