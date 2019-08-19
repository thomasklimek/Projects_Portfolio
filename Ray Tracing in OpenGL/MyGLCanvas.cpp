#define NUM_OPENGL_LIGHTS 8

#include "MyGLCanvas.h"
#include <math.h>

int Shape::m_segmentsX;
int Shape::m_segmentsY;
std::vector<SceneNode*> tree;

MyGLCanvas::MyGLCanvas(int x, int y, int w, int h, const char *l) : Fl_Gl_Window(x, y, w, h, l) {
	mode(FL_RGB | FL_ALPHA | FL_DEPTH | FL_DOUBLE);
	
	rotVec = glm::vec3(0.0f, 0.0f, 0.0f);
	eyePosition = glm::vec3(2.0f, 2.0f, 2.0f);

	isectOnly = 1;
	segmentsX = segmentsY = 10;
	recursion_val = 0;
	scale = 1.0f;
	parser = NULL;

	objType = SHAPE_CUBE;
	cube = new Cube();
	cylinder = new Cylinder();
	cone = new Cone();
	sphere = new Sphere();
	shape = cube;

	shape->setSegments(segmentsX, segmentsY);

	camera = new Camera();
	camera->orientLookAt(eyePosition, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
}
void parse_tree(SceneNode* node, SceneTransformation* trans) {
	std::vector<SceneNode*> child;
	child = node->children;

	glm::mat4 transMatrix;
	transMatrix = trans->matrix;

	//cout << "trans size " << node->transformations.size() << endl;

	for (int i = 0; i < node->transformations.size(); i++) {
		SceneTransformation *transform = node->transformations[i];
		if (transform->type == TRANSFORMATION_TRANSLATE){
			glm::mat4 temp = glm::translate(glm::mat4(1.0f), transform->translate);
    		transMatrix = transMatrix * temp;

		}
		else if (transform->type == TRANSFORMATION_SCALE){
			glm::mat4 temp = glm::scale(glm::mat4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1), transform->scale);
			transMatrix = transMatrix * temp;
		}
		else if (transform->type == TRANSFORMATION_ROTATE){
			glm::mat4 temp = glm::rotate(glm::mat4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1), transform->angle, transform->rotate);
			transMatrix = transMatrix * temp;
		}
		else if (transform->type == TRANSFORMATION_MATRIX){
			glm::mat4 temp = transform->matrix;
			transMatrix = transMatrix * temp;
		}
	}

	SceneNode *new_node = new SceneNode();
	SceneTransformation *new_trans = new SceneTransformation();
	new_trans->matrix = transMatrix;
	new_node->primitives = node->primitives;
	new_node->transformations.push_back(new_trans); 
	tree.push_back(new_node);

	if (child.size()!=0) {
		for (int i = 0; i < child.size(); i++) {
			parse_tree(child[i], new_trans);
		}
	}
	return;
}


MyGLCanvas::~MyGLCanvas() {
	delete cube;
	delete cylinder;
	delete cone;
	delete sphere;
	if (camera != NULL) {
		delete camera;
	}
	if (parser != NULL) {
		delete parser;
	}
	if (pixels != NULL) {
		delete pixels;
	}
}
float solveFace(glm::vec3 eyePointP, glm::vec3 rayV, int i, float n) {
    float t = (n - eyePointP[i]) / rayV[i];
    glm::vec3 intersect = eyePointP + rayV * t;
    if ((intersect[(i + 1) % 3] < 0.5 && intersect[(i + 1) % 3] > -0.5) &&
        (intersect[(i + 2) % 3] < 0.5 && intersect[(i + 2) % 3] > -0.5)) {
        return t;
    } else {
        return -1;
    }
}
float minP(float x, float y) {
    if (x < 0) {
        if (y < 0)
            return -1;
        else 
            return y;
    } else {
        if (y < 0)
            return x;
        else 
            return x < y ? x : y;
    }
}



double MyGLCanvas::intersect(glm::vec3 eyePointP, glm::vec3 rayV, glm::mat4 transformMatrix, OBJ_TYPE type) {	

	glm::vec3 eyePointP1 = glm::inverse(transformMatrix) * glm::vec4(eyePointP, 1.0f);
	glm::vec3 rayV1 = glm::inverse(transformMatrix) * glm::vec4(rayV, 0.0f);	
	
	if (type == SHAPE_SPHERE){

		float A, B, C, D;
		double t = -1;	
		A = glm::dot(rayV1, rayV1);
		B = 2.0f * glm::dot(eyePointP1, rayV1);
		C = glm::dot(eyePointP1, eyePointP1) - pow(0.5f, 2.0f);
		D = pow(B, 2.0f) - 4.0f * A * C;
		if (D > 0) {
        	double t0 = (-1.0*B + sqrt(D)) / (2.0*A);
        	double t1 = (-1.0*B - sqrt(D)) / (2.0*A);
       		return minP(t0, t1);
    	} else if (D == 0) {
        	double t = (-B + sqrt(D)) / (2.0*A);

        	return t;
    	} else {
        	return -1; 
    	}   
	}
	else if (type == SHAPE_CUBE) {
		float t1 = solveFace(eyePointP1, rayV1, 0, 0.5);
    	float t2 = solveFace(eyePointP1, rayV1, 1, 0.5);
    	float t3 = solveFace(eyePointP1, rayV1, 2, 0.5);
    	float t4 = solveFace(eyePointP1, rayV1, 0, -0.5);
    	float t5 = solveFace(eyePointP1, rayV1, 1, -0.5);
    	float t6 = solveFace(eyePointP1, rayV1, 2, -0.5);
    
    	return minP(t1, (minP(t2, (minP(t3, (minP(t4, (minP(t5, t6)))))))));
	}
	else if (type == SHAPE_CYLINDER){
		float t_body = -1;
		double A = rayV1[0]*rayV1[0] + rayV1[2]*rayV1[2];
    	double B = 2*eyePointP1[0]*rayV1[0] + 2*eyePointP1[2]*rayV1[2];
    	double C = eyePointP1[0]*eyePointP1[0] + eyePointP1[2]*eyePointP1[2] - 0.25;
    	double D = pow(B, 2.0f) - 4.0f * A * C;
		if (D > 0) {
        	double t0 = (-1.0*B + sqrt(D)) / (2.0*A);
        	double t1 = (-1.0*B - sqrt(D)) / (2.0*A);
       		t_body = minP(t0, t1);
    	} else if (D == 0) {
        	double t = (-B + sqrt(D)) / (2.0*A);

        	t_body = t;
        }
        glm::vec3 intersect = eyePointP1 + rayV1 * t_body;
    	if (!(intersect[1] > -0.5 && intersect[1] < 0.5)) {
        	t_body = -1;
    	}

    	float t_cap1 = (0.5 - eyePointP1[1]) / rayV1[1];
    	intersect = eyePointP1 + rayV1 * t_cap1;
    	if (!(intersect[0]*intersect[0] + intersect[2]*intersect[2] <= 0.25)) {
        	t_cap1 = -1;
    	}

    	float t_cap2 = (-0.5 - eyePointP1[1]) / rayV1[1];
    	intersect = eyePointP1 + rayV1 * t_cap2;
    	if (!(intersect[0]*intersect[0] + intersect[2]*intersect[2] <= 0.25)) {
        	t_cap2 = -1;
   		}
    
    	return minP(t_body, minP(t_cap1, t_cap2));

	}
	else if (type == SHAPE_CONE){
		float t_body = -1;
		double A = rayV1[0]*rayV1[0] + rayV1[2]*rayV1[2] - (.25*rayV1[1]*rayV1[1]);
    	double B = 2*eyePointP1[0]*rayV1[0] + 2*eyePointP1[2]*rayV1[2] - .5*eyePointP1[1]*rayV1[1] + .25*rayV1[1];
    	double C = eyePointP1[0]*eyePointP1[0] + eyePointP1[2]*eyePointP1[2] - .25*eyePointP1[1]*eyePointP1[1] + .25*eyePointP1[1] - .25*.25;

    	double D = pow(B, 2.0f) - 4.0f * A * C;
		if (D > 0) {
        	double t0 = (-1.0*B + sqrt(D)) / (2.0*A);
        	double t1 = (-1.0*B - sqrt(D)) / (2.0*A);
       		t_body = minP(t0, t1);
    	} else if (D == 0) {
        	double t = (-B + sqrt(D)) / (2.0*A);

        	t_body = t;
        }
    
    	glm::vec3 intersect = eyePointP1 + rayV1 * t_body;
    	if (!(intersect[1] > -0.5 && intersect[1] < 0.5)) {
        	t_body = -1;
    	}

    	float t_cap = (-0.5 - eyePointP1[1]) / rayV1[1];
    	intersect = eyePointP1 + rayV1 * t_cap;
    	if (!(intersect[0]*intersect[0] + intersect[2]*intersect[2] <= 0.25)) {
       		t_cap = -1;
    	}
    
    	return minP(t_body, t_cap);
	}
}

float MyGLCanvas::getNearestIntersect(glm::vec3 rayV, glm::vec3 eyePointP){
	//glm::vec3 eyePointP = camera->getEyePoint();
	//glm::vec3 rayV;
	float t = -1.0f;
	std::vector<struct RenderedPrimitive> v = parser->getFlattenedTree();
	struct RenderedPrimitive p;
	for (int k = 0; k < v.size(); k++) {
		p = v[k];
		
		t = intersect(eyePointP, rayV, p.modelView, p.primitive->type);
	}
	return t;

}
glm::vec3 computeNormal(glm::vec3 intersection, OBJ_TYPE shape) {
	double EPSILON = 0.00001;
	glm::vec3 normal = glm::vec3(0.0, 0.0, 0.0);
    switch(shape) {
        case SHAPE_CUBE: 
            if(fabs(intersection[0] + 0.5) < EPSILON)
            	normal[0] = -1;
    		if(fabs(intersection[0] - 0.5) < EPSILON)
    			normal[0] = 1;
    		if(fabs(intersection[1] + 0.5) < EPSILON)
    			normal[1] = -1;
    		if(fabs(intersection[1] - 0.5) < EPSILON)
    			normal[1] = 1;
    		if(fabs(intersection[2] + 0.5) < EPSILON)
    			normal[2] = -1;
    		if(fabs(intersection[2] - 0.5) < EPSILON)
    			normal[2] = 1;
    		return normal;
            break;
        case SHAPE_CYLINDER:
           if (IN_RANGE(intersection[1], 0.5))
               return glm::vec3(0, 1, 0);
           if (IN_RANGE(intersection[1], -0.5))
               return glm::vec3(0, -1, 0);
           return glm::vec3(intersection[0], 0, intersection[2]);
           break; 
        case SHAPE_CONE: {
           if (IN_RANGE(intersection[1], -0.5))
               return glm::vec3(0.0f, -1.0f, 0.0f);
           glm::vec3 v1 = glm::vec3(intersection[0], 0.0f, intersection[2]);
           v1 = glm::normalize(v1);
           v1[1] = 0.5f;
           return v1; }
           //v1 = glm::normalize(v1);
           break; 
        case SHAPE_SPHERE:
           return intersection;
           break;
    }
}



glm::vec3 MyGLCanvas::generateRay(int pixelX, int pixelY, glm::mat4 camToWorld) {	
	float x = -1.0f + (2.0f * pixelX) / (float)camera->getScreenWidth();
	float y = -1.0f + (2.0f * pixelY) / (float)camera->getScreenHeight();
	float z = -1.0f;	
	glm::vec3 world_space_pt = camToWorld * glm::vec4(x, y, z, 1.0f);
	glm::vec3 eyeTranslation = camToWorld * glm::vec4(0.0, 0.0, 0.0, 1.0);
	return glm::normalize(world_space_pt - eyeTranslation);
}

SceneColor getTextureColor(SceneFileMap* tex, glm::vec3 intersect){
	char* texture = tex->data->getPixels();
	int width = tex->data->getWidth();
	int height = tex->data->getHeight();
	int s = (int)(tex->repeatU * intersect[0] * width) % width;
    int t = (int)(tex->repeatV * intersect[1] * height) % height;
    int pix = t*width + s;

    SceneColor ret;
    ret.r = ((float)((unsigned char)texture[pix*3]))/ 255.0f;
    ret.g = ((float)((unsigned char)texture[pix*3+1]))/ 255.0f;
    ret.b = ((float)((unsigned char)texture[pix*3+2]))/ 255.0f;
    return ret;

}

void MyGLCanvas::renderShape(OBJ_TYPE type) {
	objType = type;
	switch (type) {
	case SHAPE_CUBE:
		shape = cube;
		break;
	case SHAPE_CYLINDER:
		shape = cylinder;
		break;
	case SHAPE_CONE:
		shape = cone;
		break;
	case SHAPE_SPHERE:
		shape = sphere;
		break;
	case SHAPE_SPECIAL1:
	default:
		shape = cube;
	}

	shape->setSegments(segmentsX, segmentsY);
	shape->draw();
}

void MyGLCanvas::loadSceneFile(const char* filenamePath) {
	if (parser != NULL) {
		delete parser;
	}
	parser = new SceneParser(filenamePath);

	bool success = parser->parse();
	cout << "success? " << success << endl;
	if (success == false) {
		delete parser;
		parser = NULL;
	}
	else {
		SceneTransformation *t = new SceneTransformation();
		t->matrix = glm::mat4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
		SceneNode* root = parser->getRootNode();
		parse_tree(root, t);
		SceneCameraData cameraData;
		parser->getCameraData(cameraData);

		camera->reset();
		camera->setViewAngle(cameraData.heightAngle);
		if (cameraData.isDir == true) {
			camera->orientLookVec(cameraData.pos, cameraData.look, cameraData.up);
		}
		else {
			camera->orientLookAt(cameraData.pos, cameraData.lookAt, cameraData.up);
		}
	}
}


void MyGLCanvas::setSegments() {
	shape->setSegments(segmentsX, segmentsY);
}

void MyGLCanvas::draw() {
	if (!valid()) {  //this is called when the GL canvas is set up for the first time or when it is resized...
		printf("establishing GL context\n");

		glViewport(0, 0, w(), h());
		updateCamera(w(), h());

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (parser == NULL) {
		return;
	}

	if (pixels == NULL) {
		return;
	}

	//this just draws the "pixels" to the screen
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glDrawPixels(w(), h(), GL_RGB, GL_UNSIGNED_BYTE, pixels);
}

int MyGLCanvas::handle(int e) {
	//printf("Event was %s (%d)\n", fl_eventnames[e], e);
	switch (e) {
	case FL_KEYUP:
		printf("keyboard event: key pressed: %c\n", Fl::event_key());
		break;
	case FL_MOUSEWHEEL:
		break;
	}

	return Fl_Gl_Window::handle(e);
}

void MyGLCanvas::resize(int x, int y, int w, int h) {
	Fl_Gl_Window::resize(x, y, w, h);
	if (camera != NULL) {
		camera->setScreenSize(w, h);
	}
	puts("resize called");
}


void MyGLCanvas::updateCamera(int width, int height) {
	float xy_aspect;
	xy_aspect = (float)width / (float)height;

	camera->setScreenSize(width, height);
}

//Given the pixel (x, y) position, set its color to (r, g, b)
void MyGLCanvas::setpixel(GLubyte* buf, int x, int y, int r, int g, int b) {
	int pixelWidth = camera->getScreenWidth();
	buf[(y*pixelWidth + x) * 3 + 0] = (GLubyte)r;
	buf[(y*pixelWidth + x) * 3 + 1] = (GLubyte)g;
	buf[(y*pixelWidth + x) * 3 + 2] = (GLubyte)b;
}

glm::vec3 MyGLCanvas::getUnit(glm::vec3 eye, glm::vec3 ray, RenderedPrimitive p, float t){
	OBJ_TYPE type = p.primitive->type;
	glm::vec3 rayObj = glm::inverse(p.modelView) * glm::vec4(ray, 0.0f);	
	glm::vec3 eyeObj = glm::inverse(p.modelView) * glm::vec4(eye, 1.0f);
	glm::vec3 coords = glm::vec3();
	double epsilon = 0.00001;

	glm::vec3 intersect = eyeObj + (rayObj * t);

	if (type == SHAPE_CUBE){

    if(fabs(intersect[0] + 0.5) < epsilon)return glm::vec3(intersect[2] + 0.5, -intersect[1] + 0.5, 0.0);
    if(fabs(intersect[0] - 0.5) < epsilon)return glm::vec3(-intersect[2] + 0.5, -intersect[1] + 0.5, 0.0);
    if(fabs(intersect[1] + 0.5) < epsilon)return glm::vec3(intersect[0] + 0.5, -intersect[2] + 0.5, 0.0);
    if(fabs(intersect[1] - 0.5) < epsilon)return glm::vec3(-intersect[0] + 0.5, -intersect[2] + 0.5, 0.0);
    if(fabs(intersect[2] + 0.5) < epsilon)return glm::vec3(-intersect[0] + 0.5, -intersect[1] + 0.5, 0.0);
    if(fabs(intersect[2] - 0.5) < epsilon)return glm::vec3(intersect[0] + 0.5, -intersect[1] + 0.5, 0.0);

    return glm::vec3(0.0, 0.0, 0.0);
	}
	else if (type == SHAPE_SPHERE){

    
    
    coords[0] = -atan2(intersect[2], intersect[0]) / (2*PI) + 0.5;
    double phi = asin(2.0*intersect[1]);
    coords[1] = -(phi / PI) + 0.5;
    coords[2] = 0.0;
    
    return coords;
	}
	else if (type == SHAPE_CYLINDER){

    if(fabs(intersect[1] + 0.5) < epsilon) 
        return glm::vec3(intersect[0] + 0.5, intersect[2] + 0.5, 0.0);
    if(fabs(intersect[1] - 0.5) < epsilon)
        return glm::vec3(intersect[0] + 0.5, intersect[2] + 0.5, 0.0);

    coords[0] = -atan2(intersect[2], intersect[0]) / (2*PI) + 0.5;
    coords[1] = -intersect[1] + 0.5;
    coords[2] = 0.0;

    return coords;
	}
	else if (type == SHAPE_CONE){
    
    	//cap
    	if(fabs(intersect[1] + 0.5) < epsilon)
        return glm::vec3(intersect[0] + 0.5, intersect[2] + 0.5, 0.0);
    
    	//body

    	coords[0] = -atan2(intersect[2], intersect[0]) / (2*PI) + 0.5;
    	coords[1] = -intersect[1] + 0.5;
    	coords[2] = 0.0;

    	return coords;
	}

}




SceneColor MyGLCanvas::recursiveRayTrace(glm::vec3 eyePointP, glm::vec3 rayV, int recursion){
	SceneColor color;
	SceneGlobalData globals;
    parser->getGlobalData(globals);
    float ka = globals.ka;
	float kd = globals.kd;
	float ks = globals.ks;




	rayV = glm::normalize(rayV);


	std::vector<struct RenderedPrimitive> v = parser->getFlattenedTree();
	struct RenderedPrimitive p;
	struct RenderedPrimitive closest;
	

	float t = 1000000;
    float tempT;
    bool found = false;

	for (int k = 0; k < v.size(); k++) {

		p = v[k];
		tempT = intersect(eyePointP, rayV, p.modelView, p.primitive->type);
		if((tempT >= 0.0) && (tempT < t)) {
			 t = tempT;
             closest = p;
             found = true;
        }
	}

	if(found){
        //cout << "eyeWorld: " << eyePointP.x << " " << eyePointP.y << " "<< eyePointP.z << endl;

        glm::vec3 objEye = glm::inverse(closest.modelView) * glm::vec4(eyePointP, 1.0f);
		glm::vec3 objRay = glm::inverse(closest.modelView) * glm::vec4(rayV, 0.0f);	
        glm::vec3 intersection_obj = t * objRay + objEye;
        glm::vec3 normal = computeNormal(intersection_obj, closest.primitive->type); 
        //normal = glm::normalize(normal);         
        //normal = glm::transpose(glm::inverse(p.modelView)) * glm::vec4(normal, 1.0f);
        normal = glm::normalize(normal);




        SceneColor Oa = closest.primitive->material.cAmbient;

		//cout << Oa.r << " " << Oa.g << " " << Oa.b << endl;
		SceneColor Od = closest.primitive->material.cDiffuse;
		SceneColor Os = closest.primitive->material.cSpecular;
		float f = closest.primitive->material.shininess;

        glm::vec3 lightDir;
        float lightDist;
        glm::vec3 reflectiveRay;
        //color = color + (closest.primitive->material.cAmbient * globals.ka);
        color = Oa * ka;
        
        SceneColor diffConst = closest.primitive->material.cDiffuse * globals.kd;
        SceneColor diffColor = SceneColor();
        //cout << "diffColor" << diffColor.r << diffColor.g << diffColor.b << endl;


        SceneColor specConst = closest.primitive->material.cSpecular * globals.ks;
        SceneColor specColor = SceneColor();

        float specularF = closest.primitive->material.shininess;

        SceneColor reflectiveConst = closest.primitive->material.cReflective * ks;
        SceneColor reflectiveColor = SceneColor();


        //glm::vec3 hitPoint = eyePointP + (rayV * t);// - (rayV * 0.00001f);
        glm::vec3 hitPoint = closest.modelView * glm::vec4(intersection_obj, 1.0f);
        hitPoint = hitPoint + normal * 0.0001f;
        SceneColor li;



        for(int l = 0; l < parser->getNumLights() ; l++){
        	SceneLightData light;

        	//cout << "light loop" <<endl;
            parser->getLightData(l, light);
            if(light.type == LIGHT_DIRECTIONAL){
            	//cout << "direction light #" << l << " : "  << light.dir.x << light.dir.y << light.dir.z <<  endl;
                lightDir = -light.dir;//back towards where the light is
                lightDist = 1e9;
            }
            else if(light.type == LIGHT_POINT){
            	//cout << "lightpoints" << endl;
                lightDir = light.pos - hitPoint;
                lightDist = glm::length(lightDir);
        	}
        	lightDir = glm::normalize(lightDir);

        	li = light.color;

            int blocked = 0;
            //float dist = 0;
            tempT = 0;
            for (int m = 0; m < v.size(); m++) {
				p = v[m];
				tempT = intersect(hitPoint, lightDir, p.modelView, p.primitive->type);
                if(tempT >  0.00001)
                	blocked = 1;

            }

            if(!blocked){

            	
                float dotProd = glm::dot(lightDir, normal);
                //if (closest.primitive->type == SHAPE_SPHERE && dotProd < 0 && l == 0){
                	//cout << "light dir: " << lightDir.x  << " " << lightDir.y << " " << lightDir.z << endl;
            		//cout << "normal: " << normal.x << " " << normal.y << " " << normal.z << endl;
                	//cout << "dot prod: "<< dotProd << endl;
                //}
                if(dotProd < 0) dotProd = 0.0f;

                SceneColor contrib = li * kd * Od * dotProd;
                diffColor = diffColor + contrib;
                 if (closest.primitive->type == SHAPE_SPHERE && l == 0){
                //cout << "diffColor" << diffColor.r << diffColor.g << diffColor.b << endl;
                }

                reflectiveRay = (2 * dotProd * normal) - lightDir;
                reflectiveRay = glm::normalize(reflectiveRay);
                dotProd = glm::dot(reflectiveRay,-rayV);
                if(dotProd < 0.0f)
                	dotProd = 0.0f;
                contrib = specConst * (light.color * pow(dotProd,specularF));

                specColor = specColor + contrib;
            }
        }

        if(recursion > 0){
            float dotProd = glm::dot(rayV, normal);
            reflectiveRay = rayV - (2 * dotProd * normal);
            reflectiveColor = reflectiveConst * recursiveRayTrace(hitPoint, reflectiveRay, recursion-1);
        }
        if(closest.primitive->material.blend > 1e-10){
        	SceneFileMap *tex = closest.primitive->material.textureMap;
        	if (tex->isUsed) {
				SceneColor texcolor;
				glm::vec3 unitIntersection = getUnit(eyePointP, rayV, closest, t);
				texcolor = getTextureColor(tex, unitIntersection);
            	float blend = closest.primitive->material.blend;
            	diffColor = diffColor * (1.0-blend) + texcolor * (blend);
            }
        }
        color = color + diffColor + specColor + reflectiveColor;
	}

	return color;
}



















void MyGLCanvas::renderScene() {
	cout << "render button clicked!" << endl;

	if (parser == NULL) {
		cout << "no scene loaded yet" << endl;
		return;
	}

	int pixelWidth = w();
	int pixelHeight = h();


	updateCamera(pixelWidth, pixelHeight);

	if (pixels != NULL) {
		delete pixels;
	}
	pixels = new GLubyte[pixelWidth  * pixelHeight * 3];
	memset(pixels, 0, pixelWidth  * pixelHeight * 3);
	glm::vec3 eyePointP = camera->getEyePoint();
	glm::vec3 rayV;
	std::vector<struct RenderedPrimitive> v = parser->getFlattenedTree();
	struct RenderedPrimitive p;
	glm::mat4 camToWorld = camera->getInverseModelViewMatrix() * camera->getInverseScaleMatrix();
	int k_last = -1;
	//float t_min = -1;
	SceneColor color;


	for (int i = 0; i < pixelWidth; i++) {
		for (int j = 0; j < pixelHeight; j++) {
			//TODO: this is where your ray casting will happen!
			rayV = generateRay(i, j, camToWorld);
			if (isectOnly == 1) {
				for (int k = 0; k < v.size(); k++) {
					p = v[k];
					float t = 0.0;
					t = intersect(eyePointP, rayV, p.modelView, p.primitive->type);
					if (t > 0.0){
						setpixel(pixels, i, j, 255, 255, 255);
					}
				}
			} else {
				color = recursiveRayTrace(eyePointP, rayV, recursion_val);
				color = color * 255.0;
            	if(color.r > 255)color.r = 255;
            	if(color.g > 255)color.g = 255;
            	if(color.b > 255)color.b = 255;
				setpixel(pixels, i, j, color.r, color.g, color.b);
				
				/*
				float t_min = -1;
				for (int k = 0; k < v.size(); k++) {
					p = v[k];
					float t = 0.0;
					t = intersect(eyePointP, rayV, p.modelView, p.primitive->type);
					if (t >= 0 && (t_min < 0 || t < t_min)) {


						glm::vec3 eyePointP1 = glm::inverse(p.modelView) * glm::vec4(camera->getEyePoint(), 1.0f);
						glm::vec3 rayV1 = glm::inverse(p.modelView) * glm::vec4(rayV, 0.0f);	
                		t_min = t; 
                		glm::vec3 intersection_obj = (t_min - 0.0001f) * rayV1 + eyePointP1;
                		glm::vec3 normal = computeNormal(intersection_obj, p.primitive->type); 
                		normal = glm::transpose(glm::inverse(p.modelView)) * glm::vec4(normal, 1.0f);
                		glm::vec3 intersection_world = p.modelView * glm::vec4(intersection_obj, 1.0);
                		normal = glm::normalize(normal);


                		color = getColor(p.primitive->material, normal, intersection_world, rayV, p, t);
                		setpixel(pixels, i, j, color.r, color.g, color.b);
            		}
				}
				*/
			}
		}
	}
	cout << "render complete" << endl;
	redraw();
}