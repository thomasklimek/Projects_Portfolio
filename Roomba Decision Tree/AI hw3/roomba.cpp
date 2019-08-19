/*	Thomas Klimek
	Comp 131 AI
	Assignment #3
*/

// To Run this code: g++ roomba.cpp -o roomba
//					 ./roomba


#include <stdlib.h>
#include <stdio.h>
#include <iostream> // cout

// --------------------------------------EXPLANATION-------------------------------------------
/* This assignment practices implementing a behavior tree for the AI of a roomba vaccuum cleaner.
 
 * I have implemented the tree from the assignment PDF in the function behavior_tree represented as 
 * nested if-statements. 

 * I implemented the blackboard as a struct which serves as an abstraction for a hash-table.

 * I implemented the tasks from the behavior tree as functions to return the task status. This
 * abstraction allows us to focus on the behavior rather than the tasks. It also allows for easy
 * integration of the behavior tree to real tasks, by changing the task functions and leaving the 
 * behavior_tree function in tact.

 */

using namespace std;

// --------------------------------------BLACKBOARD-------------------------------------------
// This struct serves as our blackboard object: It can be conceptualized as a hash table where
// the member name (i.e. SPOT, GENERAL) is the key and what is stored in that member is the value.
// Thus the abstraction to use a blackboard as a hashtable would be blackboard.KEY = VALUE
// for example blackboard.SPOT = true -- (SPOT is the key and true is the value)
typedef struct blackboard {
	int BATTERY_LEVEL; //: an integer number between 0 and 100 representing % of battery charge
	bool SPOT; // a Boolean value – TRUE if the command was requested, FALSE otherwise 
	bool GENERAL; // a Boolean value – TRUE if the command was requested, FALSE otherwise 
	bool DUSTY_SPOT; // a Boolean value – TRUE if the sensor detected a dusty spot during the 
					 // cycle, FALSE otherwise 
	int HOME_PATH; // The path to the docking station -- represented as an integer distance
				   // because we are considering a roomba in 1 dimention
} blackboard;

void initialize_blackboard(blackboard &b){
	cout << "Please enter initial battery level (0-100): " << endl;
	cin >> b.BATTERY_LEVEL;

	cout << "Please enter spot command: false or true (0 or 1): " << endl;
	cin >> b.SPOT;

	cout << "Please enter general command: false or true (0 or 1): " << endl;
	cin >> b.GENERAL;

	cout << "Please enter dusty spot command: false or true (0 or 1): " << endl;
	cin >> b.DUSTY_SPOT;

	cout << "Please enter initial homepath distance (int): " << endl;
	cin >> b.HOME_PATH;

}


// --------------------------------------TASK FUNCTIONS----------------------------------------
// Here are the functions represented as task nodes in the diagram for the behavior tree. Most of
// the tasks have been simplified to simply output the task name and status, so we can focus 
// on the results of the behavior tree rather than the implementation of the tasks. If the behavior
// tree where to be implemeneted on a real roomba, these task functions would be changed to interface
// with the roomba.

// DESIGN NOTE: This design would be cleaner if the blackboard was implemented as a class, rather than
// a struct. Then we could have these as member functions of the class, and a blackboard would not need
// to be passed to each function. For simplicity I have used structs to represent the blackboard, but a
// modification would be made if this were production code to use classes and inheritance. 

// FIND_HOME is the task for finding home, because we are not dealing with a actual 2D or 3D space, we
// simply return that the task has been complete when it is called in the behavior tree
void FIND_HOME(blackboard &b){
	cout << "FIND_HOME: COMPLETED" << endl;
}

// GO_HOME is the task to send the roomba home. For this function I have made the assumption that the
// roomba operates in a 1D space. Thus the HOME_PATH is represented as an integer distance and decrements
// as the roomba returns home. Then it is set to 0 when the roomba arrives home.
void GO_HOME(blackboard &b){
	for (int i = 0; i< b.HOME_PATH; i++){
		cout << "GO_HOME: RUNNING " << i << endl;
	}
	b.HOME_PATH = 0;
	cout << "GO_HOME: COMPLETED" << endl;
}

// DOCK is the task to dock the roomba to the charging point once it has returned home. In this task
// the BATTERY_LEVEL is set to 100 to show that the roomba has been charged.
void DOCK(blackboard &b){
	b.BATTERY_LEVEL = 100;
	cout << "DOCK: COMPLETED" << endl;
}

// CLEAN_SPOT is the task to clean a spot for a certain amount of time. This time depends on a dusty
// or dirty spot which is handled in the behavior tree. For this task, the assumption is made that the
// roomba loses 1% of its battery for every second of cleaning, thus the battery level is decremented by
// the argument cleantime.
void CLEAN_SPOT(blackboard &b, int cleantime) {
	cout << "CLEAN_SPOT: RUNNING " << cleantime << " seconds" << endl;
	b.BATTERY_LEVEL -= cleantime;
	cout << "CLEAN_SPOT: COMPLETED" << endl;
}

// CLEAN is the general cleaning task. For this task two assumptions are made. 
// 1. Each time clean is called, the roomba advances to a new space to clean, increasing
//	  the HOME_PATH path distance by 1.
// 2. Each time clean is called, the roomba battery loses 10% of its charge.
void CLEAN(blackboard &b){
	b.HOME_PATH++;
	b.BATTERY_LEVEL -= 10;
	cout << "CLEAN: COMPLETED" << endl;
}

// DONE_SPOT is the task called after a spot is done cleaning. It simply sets the SPOT
// key of the blackboard to false.
void DONE_SPOT (blackboard &b){
	b.SPOT = false;
	cout << "DONE_SPOT: COMPLETED" << endl;
}

// DONE_GENERAL is the task called after general cleaning has been completed. It simply
// sets the GENERAL key of the blackboard to false.
void DONE_GENERAL (blackboard &b){
	b.GENERAL = false;
	cout << "DONE_GENERAL: COMPLETED" << endl;
}

// DO_NOTHING is the roomba idle task.
void DO_NOTHING (blackboard &b){
	cout << "DO_NOTHING: COMPLETED" << endl;
}

// --------------------------------------BEHAVIOR TREE----------------------------------------
// This function serves as the behavior tree and the brain for the roomba. It represents the tree
// from the assignment PDF as nested IF statements. 

// behavior_tree accepts a blackboard object as an argument, and based on the contents of the
// blackboard calls different task functions for the roomba and updates the blackboard accordingly. 
// One call of behavior_tree represents one cycle of deision making by the roomba. This function is
// called constantly in a real roomba.
void behavior_tree(blackboard &b){

	// first node in behavior tree
	if (b.BATTERY_LEVEL < 30) {
		FIND_HOME(b);
		GO_HOME(b);
		DOCK(b);

	} 

	// second node in behavior tree
	if (b.SPOT){
		CLEAN_SPOT(b, 20);
		DONE_SPOT(b);
	} 
	if (b.GENERAL){
		do {
			if (b.DUSTY_SPOT){
				CLEAN_SPOT(b, 35);
				b.DUSTY_SPOT = false;
			}
			CLEAN(b);
		} while (b.BATTERY_LEVEL > 30);
		DONE_GENERAL(b);
	}

	// third node in behavior tree
	DO_NOTHING(b);

}


int main(){
	blackboard b;

	// defualt initialize blackboard
	b.BATTERY_LEVEL = 20;
	b.SPOT = true;
	b.GENERAL = true;
	b.DUSTY_SPOT = true;
	b.HOME_PATH = 5;

	initialize_blackboard(b);

	behavior_tree(b);


	return 0;

}