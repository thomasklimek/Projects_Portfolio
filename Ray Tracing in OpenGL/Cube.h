#ifndef CUBE_H
#define CUBE_H

#include "Shape.h"
#include <vector>

using namespace std;

class Cube : public Shape {
public:
	Cube() {lastX = -1; lastY = -1;};
	~Cube() {};
	int lastX;
	int lastY;

	vector<vector<float> > triangles;
	vector<vector<float> > normals;

	OBJ_TYPE getType() {
		return SHAPE_CUBE;
	}

	void draw() {
		if (lastX != m_segmentsX || lastY != m_segmentsY){
			lastX = m_segmentsX;
			lastY = m_segmentsY;
			float xLength = 1.0f / m_segmentsX;
			float yLength = 1.0f / m_segmentsY;

			triangles.clear();
			normals.clear();

			// front side
			for (int x = 0; x < m_segmentsX; x++){
				for (int y = 0; y < m_segmentsY; y++){
					vector<float> normal;
					normal.push_back(0.0f);
					normal.push_back(0.0f);
					normal.push_back(1.0f);

					float x1 = x * xLength - 0.5f;
					float x2 = (x + 1) * xLength - 0.5f;
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;
					float z = 0.5f;

					vector<float> p1, p2, p3, p4;

					p1.push_back(x1);
					p1.push_back(y1);
					p1.push_back(z);

					p2.push_back(x2);
					p2.push_back(y1);
					p2.push_back(z);

					p3.push_back(x1);
					p3.push_back(y2);
					p3.push_back(z);

					p4.push_back(x2);
					p4.push_back(y2);
					p4.push_back(z);

					triangles.push_back(p1);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p3);
					normals.push_back(normal);

					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p4);
					normals.push_back(normal);
					triangles.push_back(p3);
					normals.push_back(normal);
				}
			}

			// back side
			for (int x = 0; x < m_segmentsX; x++){
				for (int y = 0; y < m_segmentsY; y++){
					vector<float> normal;
					normal.push_back(0.0f);
					normal.push_back(0.0f);
					normal.push_back(-1.0f);

					float x1 = x * xLength - 0.5f;
					float x2 = (x + 1) * xLength - 0.5f;
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;
					float z = -0.5f;

					vector<float> p1, p2, p3, p4;

					p1.push_back(x1);
					p1.push_back(y1);
					p1.push_back(z);

					p2.push_back(x2);
					p2.push_back(y1);
					p2.push_back(z);

					p3.push_back(x1);
					p3.push_back(y2);
					p3.push_back(z);

					p4.push_back(x2);
					p4.push_back(y2);
					p4.push_back(z);

					triangles.push_back(p3);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p1);
					normals.push_back(normal);

					triangles.push_back(p3);
					normals.push_back(normal);
					triangles.push_back(p4);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
				}
			}

			// right side
			for (int x = 0; x < m_segmentsX; x++){
				for (int y = 0; y < m_segmentsY; y++){
					vector<float> normal;
					normal.push_back(1.0f);
					normal.push_back(0.0f);
					normal.push_back(0.0f);

					float x1 = x * xLength - 0.5f;
					float x2 = (x + 1) * xLength - 0.5f;
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;
					float z = 0.5f;

					vector<float> p1, p2, p3, p4;

					p1.push_back(z);
					p1.push_back(x1);
					p1.push_back(y1);

					p2.push_back(z);
					p2.push_back(x2);
					p2.push_back(y1);
					
					p3.push_back(z);
					p3.push_back(x1);
					p3.push_back(y2);
					
					p4.push_back(z);
					p4.push_back(x2);
					p4.push_back(y2);
					
					triangles.push_back(p1);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p3);
					normals.push_back(normal);

					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p4);
					normals.push_back(normal);
					triangles.push_back(p3);
					normals.push_back(normal);
				}
			}

			// left side
			for (int x = 0; x < m_segmentsX; x++){
				for (int y = 0; y < m_segmentsY; y++){
					vector<float> normal;
					normal.push_back(-1.0f);
					normal.push_back(0.0f);
					normal.push_back(0.0f);

					float x1 = x * xLength - 0.5f;
					float x2 = (x + 1) * xLength - 0.5f;
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;
					float z = -0.5f;

					vector<float> p1, p2, p3, p4;

					p1.push_back(z);
					p1.push_back(x1);
					p1.push_back(y1);
					
					p2.push_back(z);
					p2.push_back(x2);
					p2.push_back(y1);
					
					p3.push_back(z);
					p3.push_back(x1);
					p3.push_back(y2);
					
					p4.push_back(z);
					p4.push_back(x2);
					p4.push_back(y2);

					triangles.push_back(p3);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p1);
					normals.push_back(normal);

					triangles.push_back(p3);
					normals.push_back(normal);
					triangles.push_back(p4);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
				}
			}

			// top side
			for (int x = 0; x < m_segmentsX; x++){
				for (int y = 0; y < m_segmentsY; y++){
					vector<float> normal;
					normal.push_back(0.0f);
					normal.push_back(1.0f);
					normal.push_back(0.0f);

					float x1 = x * xLength - 0.5f;
					float x2 = (x + 1) * xLength - 0.5f;
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;
					float z = 0.5f;

					vector<float> p1, p2, p3, p4;

					p1.push_back(y1);
					p1.push_back(z);
					p1.push_back(x1);
					
					p2.push_back(y1);
					p2.push_back(z);
					p2.push_back(x2);
					
					p3.push_back(y2);
					p3.push_back(z);
					p3.push_back(x1);
					
					p4.push_back(y2);
					p4.push_back(z);
					p4.push_back(x2);
					
					triangles.push_back(p1);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p3);
					normals.push_back(normal);

					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p4);
					normals.push_back(normal);
					triangles.push_back(p3);
					normals.push_back(normal);
				}
			}

			// bottom side
			for (int x = 0; x < m_segmentsX; x++){
				for (int y = 0; y < m_segmentsY; y++){
					vector<float> normal;
					normal.push_back(0.0f);
					normal.push_back(-1.0f);
					normal.push_back(0.0f);

					float x1 = x * xLength - 0.5f;
					float x2 = (x + 1) * xLength - 0.5f;
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;
					float z = -0.5f;

					vector<float> p1, p2, p3, p4;

					p1.push_back(y1);
					p1.push_back(z);
					p1.push_back(x1);
					
					p2.push_back(y1);
					p2.push_back(z);
					p2.push_back(x2);
					
					p3.push_back(y2);
					p3.push_back(z);
					p3.push_back(x1);
					
					p4.push_back(y2);
					p4.push_back(z);
					p4.push_back(x2);
					
					triangles.push_back(p3);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
					triangles.push_back(p1);
					normals.push_back(normal);

					triangles.push_back(p3);
					normals.push_back(normal);
					triangles.push_back(p4);
					normals.push_back(normal);
					triangles.push_back(p2);
					normals.push_back(normal);
				}
			}
		}

		glBegin(GL_TRIANGLES);
		for (int i = 0; i < triangles.size(); i++){
			glNormal3f(normals[i][0], normals[i][1], normals[i][2]);
			glVertex3f(triangles[i][0], triangles[i][1], triangles[i][2]);
		}
		glEnd();
	};


	void drawNormal() {

		glLineWidth(2); 
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
		for (int i = 0; i < triangles.size(); i++){
			glVertex3f(triangles[i][0], triangles[i][1], triangles[i][2]);

			glVertex3f(triangles[i][0] + normals[i][0] * .1f, 
					   triangles[i][1] + normals[i][1] * .1f, 
					   triangles[i][2] + normals[i][2] * .1f);
		}
		glEnd();
		
	};

	

private:
};

#endif