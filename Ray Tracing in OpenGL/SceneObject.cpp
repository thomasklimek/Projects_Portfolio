/*  =================== File Information =================
	File Name: object.h
	Description:
	Author: Michael Shah
	Last Modified: 3/10/14

	Purpose:
	Usage:

	Further Reading resources:
	===================================================== */
#include <FL/gl.h>
#include <iostream>
#include <cmath>
#include "SceneObject.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923
	/*	===============================================
	Desc:
	Precondition:
	Postcondition:
	=============================================== */
SceneObject::SceneObject() {
	baseTexture = NULL;
	blendTexture = NULL;
}
/*	===============================================
Desc:
Precondition:
Postcondition:
=============================================== */
SceneObject::~SceneObject() {
	if (baseTexture != NULL) {
		delete baseTexture;
	}
	if (blendTexture != NULL) {
		delete blendTexture;
	}
}
/*	===============================================
Desc:
Precondition:
Postcondition:
=============================================== */
void SceneObject::paintTexture(float x, float y, char r, char g, char b) {
	int width = blendTexture->getWidth();
	int height = blendTexture->getHeight();

	float tempX = x + 0.5;
	tempX = 0.25 + (tempX*0.5);
	float u = tempX;
	float v = 1 - (y + 0.5);
	int drawX = u * width;
	int drawY = v * height;
	for (int xx = drawX - 5; xx < drawX + 5; xx++) {
		for (int yy = drawY - 5; yy < drawY + 5; yy++) {
			if ((xx > 0) && (xx < width) && (yy > 0) && (yy < height)) {
				blendTexture->setPixel(xx, yy, r, g, b);
			}
		}
	}
	glGenTextures(1, &blendTextureID);
	glBindTexture(GL_TEXTURE_2D, blendTextureID);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RGB,
		blendTexture->getWidth(),
		blendTexture->getHeight(),
		0,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		blendTexture->getPixels());

}


void SceneObject::paintTexture(glm::vec3 isect, char r, char g, char b) {
	int width = blendTexture->getWidth();
	int height = blendTexture->getHeight();

	std::cout << "isect: " << isect[0] << ", " << isect[1] << std::endl;

	float u = glm::degrees(atan2(isect[2], isect[0]));
	float v = glm::degrees(atan2(isect[2], isect[1]));

	std::cout << "u: " << u << " v: " << v << std::endl;

	u = u / 180.0f;
	u = 1.0f - u;
	v = v / 180.0f;

	u = 0.25 + (u * 0.5);

	int drawX = u * width;
	int drawY = v * height;
	for (int xx = drawX - 5; xx < drawX + 5; xx++) {
		for (int yy = drawY - 5; yy < drawY + 5; yy++) {
			if ((xx > 0) && (xx < width) && (yy > 0) && (yy < height)) {
				blendTexture->setPixel(xx, yy, r, g, b);
			}
		}
	}
	glGenTextures(1, &blendTextureID);
	glBindTexture(GL_TEXTURE_2D, blendTextureID);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RGB,
		blendTexture->getWidth(),
		blendTexture->getHeight(),
		0,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		blendTexture->getPixels());
}


/*	===============================================
Desc:	This instantiates an image to be rendered.

		textureNumber 0 corresponds to the base texture
		textureNuber 1 corresponds to the next texture

		If texture number is less than 0, then default to 0
		If texture number is greater than 1, then default to 1
Precondition:
Postcondition:
=============================================== */
void SceneObject::setTexture(int textureNumber, std::string _fileName) {
	/*
		Algorithm
		Step 1: Allocate memory for texture
		Step 2: Generate textures and bind them to object
	*/

	if (textureNumber <= 0) {
		if (baseTexture == NULL) {
			baseTexture = new ppm(_fileName);
			baseTextureID = loadTexture(baseTexture->getWidth(), baseTexture->getHeight(), baseTexture->getPixels());
		}
		else {
			delete baseTexture;
			baseTexture = new ppm(_fileName);
			glBindTexture(GL_TEXTURE_2D, 0);
			baseTextureID = loadTexture(baseTexture->getWidth(), baseTexture->getHeight(), baseTexture->getPixels());
		}
	}
	else if (textureNumber >= 1) {
		if (blendTexture == NULL) {
			blendTexture = new ppm(_fileName);
			blendTextureID = loadTexture(blendTexture->getWidth(), blendTexture->getHeight(), blendTexture->getPixels());
		}
		else {
			delete blendTexture;
			blendTexture = new ppm(_fileName);
			glBindTexture(GL_TEXTURE_2D, 0);
			blendTextureID = loadTexture(blendTexture->getWidth(), blendTexture->getHeight(), blendTexture->getPixels());
		}
	}

	// Repeat the process for the 2nd texture
}

/*	===============================================
Desc:	Loads a new array of colors of a specified dimension
		into our object.
Precondition:
Postcondition:
=============================================== */
GLuint SceneObject::loadTexture(int width, int height, char* pixels) {
	//std::cout << "loading texture" << std::endl;
	GLuint textureId;
	glGenTextures(1, &textureId);
	// Now we begin to edit the texture
//std::cout << "binding texture" << std::endl;
	glBindTexture(GL_TEXTURE_2D, textureId);
	//  Now map the actual ppm image to the texture
//std::cout << "store texture in memory " << std::endl;
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RGB,
		width,
		height,
		0,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		pixels);
	// Now at this point we have generated some textures that can be used for this object
	// We do not need our ppms at this point, because they have been shipped off to the
	// GPU!

	return textureId;
}




/*	===============================================
Desc:
Precondition:
Postcondition:
=============================================== */
void SceneObject::render() {
	//glutSolidSphere(1.0,8,8);

	// Enable the ability to draw 2D textures
	glEnable(GL_TEXTURE_2D);

	// Render the base texture for us to see how it changes
	glBindTexture(GL_TEXTURE_2D, baseTextureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glPushMatrix();
	glTranslatef(0.75f, 0.75f, -1.0f);
	glScalef(0.15, 0.15, 0.15);
	glBegin(GL_QUADS);
	glNormal3f(0.0, 1.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, 1.0f, 0.0f);
	glEnd();
	glPopMatrix();

	// Render the blend texture for us to see
	glBindTexture(GL_TEXTURE_2D, blendTextureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glPushMatrix();
	glTranslatef(-0.75f, 0.75f, -1.0f);
	glScalef(0.15, 0.15, 0.15);
	glBegin(GL_QUADS);
	glNormal3f(0.0, 1.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, 1.0f, 0.0f);
	glEnd();
	glPopMatrix();

	glDisable(GL_TEXTURE_2D);
}

/*	===============================================
Desc:	This function is an example of how to map a full
		texture to an object with an arbritrary shape.

		You can also see how two triangles make up a quad and then
		can be used to go through a surface.  This works okay,
		if and only if you have lots and lots of triangles to work with!
Precondition:
Postcondition:
=============================================== */
void SceneObject::drawTexturedSphere()
{
	float radius = 0.5;
	//float angle = 0;
	float angleH = -PI / (float)2.0;
	float angle = PI / 2.0f;

	int m_segmentsX = 100;
	int m_segmentsY = 100;

	float angle_delta = 2.0 * PI / (float)m_segmentsX;
	float angleH_delta = PI / (float)m_segmentsY;

	glEnable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, blendTextureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBegin(GL_TRIANGLES);
	for (int i = 0; i < m_segmentsY; i++) {
		angle = 0;
		for (int j = 0; j < m_segmentsX; j++) {
			// equation of a sphere
			float x = radius * cos(angleH) * cos(angle);
			float z = radius * cos(angleH) * sin(angle);
			float y = radius * sin(angleH);

			float newx = radius * cos(angleH) * cos(angle + angle_delta);
			float newz = radius * cos(angleH) * sin(angle + angle_delta);
			float newy = radius * sin(angleH);

			// Generate second triangle to make a quad
			float x_next = radius * cos(angleH + angleH_delta) * cos(angle);
			float z_next = radius * cos(angleH + angleH_delta) * sin(angle);
			float y_next = radius * sin(angleH + angleH_delta);

			float newx_next = radius * cos(angleH + angleH_delta) * cos(angle + angle_delta);
			float newz_next = radius * cos(angleH + angleH_delta) * sin(angle + angle_delta);
			float newy_next = radius * sin(angleH + angleH_delta);

			// i = m_segmentsY = 100 maximum
			// j = m_segmentsY = 100 maximum

			float textureCoordX_delta = (float)1 / (float)m_segmentsX;	// increments with j
			float textureCoordY_delta = (float)1 / (float)m_segmentsY; 	// increments with i

			// Commented out glTexCoord2f represents mapping
			// an entire texture to a set of two triangles
			// Note that the top of the sphere
			// will not quite get mapped correctly due to this algorithm

			// Note that we subtract by 1 at the beginning of tx,ty,etx, and ety to flip
			// the texture.
			float tx = 1 - j * textureCoordX_delta;// starting x pixel coordinate
			float ty = 1 - i * textureCoordY_delta;// starting y pixel coordinate
			float etx = 1 - (j + 1)*textureCoordX_delta;// ending x pixel coordinate
			float ety = 1 - (i - 1)*textureCoordY_delta;// ending y pixel coordinate

			glTexCoord2f(tx, ety); 		// glTexCoord2f(0.0f, 1.0f);
			glNormal3f(x, y, z);
			glVertex3f(x, y, z);

			glTexCoord2f(etx, ety); 		// glTexCoord2f(1.0f, 1.0f);
			glNormal3f(newx, newy, newz);
			glVertex3f(newx, newy, newz);

			glTexCoord2f(etx, ty); 		// glTexCoord2f(1.0f, 0.0f);
			glNormal3f(newx_next, newy_next, newz_next);
			glVertex3f(newx_next, newy_next, newz_next);

			glTexCoord2f(etx, ty); 		// glTexCoord2f(1.0f, 0.0f);
			glNormal3f(newx_next, newy_next, newz_next);
			glVertex3f(newx_next, newy_next, newz_next);

			glTexCoord2f(tx, ty); 		// glTexCoord2f(0.0f, 0.0f);
			glNormal3f(x_next, y_next, z_next);
			glVertex3f(x_next, y_next, z_next);

			glTexCoord2f(tx, ety); 		// glTexCoord2f(0.0f, 1.0f);
			glNormal3f(x, y, z);
			glVertex3f(x, y, z);
			angle = angle + angle_delta;
		}
		angleH = angleH + angleH_delta;
	}
	glEnd();

	glDisable(GL_TEXTURE_2D);
}
