package pacman.game;

import java.io.*;
import java.util.Random;

/**
 * This class contains the enumerations for the moves and the ghosts as well as
 * as the constants of the game. If you should change the constants, bear in mind
 * that this might significantly affect the game and hence the performance of your
 * controller. The set of constants used in the competition may be found on the
 * competition website.
 */
public class Constants
{
	/**
	 * Enumeration for the moves that are possible in the game. At each time step,
	 * a controller is required to supply one of the 5 actions available. If the 
	 * controller replies NEUTRAL or does not reply in time, the previous action is 
	 * repeated. If the previous action is not a legal move, a legal move is chosen 
	 * uniformly at random.
	 */
	public enum MOVE 
	{
		UP 		{ public MOVE opposite(){return MOVE.DOWN;		};},	
		RIGHT 	{ public MOVE opposite(){return MOVE.LEFT;		};}, 	
		DOWN 	{ public MOVE opposite(){return MOVE.UP;		};},		
		LEFT 	{ public MOVE opposite(){return MOVE.RIGHT;		};}, 	
		NEUTRAL	{ public MOVE opposite(){return MOVE.NEUTRAL;	};};	
		
		public abstract MOVE opposite();
	};
	
	/**
	 * Enumeration for the ghosts. The integer arguments are the initial lair times.
	 */
	public enum GHOST
	{
		//if using line ghost mode, remove a 0 from each one
		
		BLINKY(40),
		PINKY(60),
		INKY(80),
		SUE(100);
		
		public int initialLairTime;
		
		GHOST(int lairTime)
		{
			this.initialLairTime=lairTime;
		}
		
	};
	
	/**
	 * DM stands for Distance Metric, a simple enumeration for use with methods that 
	 * require a distance metric. The metric available are as follows:
	 * PATH: the actual path distance (i.e., number of step required to reach target)
	 * EUCLID: Euclidean distance using the nodes' x and y coordinates
	 * MANHATTAN: Manhattan distance (absolute distance between x and y coordinates)
	 */
	public enum DM {PATH, EUCLID, MANHATTAN};
	
	// Local game parameters
	public int MAZE_NUM = 1;			// Maze index for the current game
	public int GHOST_TYPE = 1;			// 0 = RandomGhosts, 1 = StandardGhosts, 2 = ChaserGhosts, 3 = LineGhosts
	public String SOURCE_TASK = "independent_1k";
	public String LEARNER = "independent";				// independent, transfer
	public int LENGTH = 1000;
	public String SAVE_DIR = null;
	
	public double PERFORMANCE_THRESHOLD = Double.POSITIVE_INFINITY;			// Stop training after this performace is reached (this value is only used in Experiments file)
	
	public long SEED;
	public  int PILL=10;						//points for a normal pill
	public  int POWER_PILL=50;					//points for a power pill
	public  int GHOST_EAT_SCORE=200;			//score for the first ghost eaten (doubles every time for the duration of a single power pill)
	public  int EDIBLE_TIME= 200;				//initial time a ghost is edible for (decreases as level number increases) 
	public  float EDIBLE_TIME_REDUCTION=0.9f;	//reduction factor by which edible time decreases as level number increases
	public  int COMMON_LAIR_TIME=40;			//time spend in lair after being eaten
	public  float LAIR_REDUCTION=0.9f;			//reduction factor by which lair times decrease as level number increases
	public  int LEVEL_LIMIT=2000;  //Integer.MAX_VALUE;//3000;			//time limit for a level
	public  float GHOST_REVERSAL=0.0f;//0.0015f;		//probability of a global ghost reversal event
	public  int MAX_LEVELS=1;//16;					//maximum number of levels played before the end of the game
	public  int EXTRA_LIFE_SCORE=Integer.MAX_VALUE;//10000;			//extra life is awarded when this many points have been collected
	public  int EAT_DISTANCE=2;					//distance in the connected graph considered close enough for an eating event to take place
	public  int NUM_GHOSTS=4;					//number of ghosts in the game
	public  int DELAY=4;						//delay (in milliseconds) between game advancements						
	public  int NUM_LIVES=1;//3;					//total number of lives Ms Pac-Man has (current + NUM_LIVES-1 spares)
	// Controls how fast ghosts move when pacman has eaten power pill. Lower numbers = slower (1 = frozen). 
	public int GHOST_SPEED_REDUCTION=2;		//difference in speed when ghosts are edible (every GHOST_SPEED_REDUCTION, a ghost remains stationary)
	public  int EDIBLE_ALERT=30;				//for display only (ghosts turning blue)
	public  int INTERVAL_WAIT=1;				//for quicker execution: check every INTERVAL_WAIT ms to see if controllers have returned
	public  float PILL_REWARD_REDUCTION=0.0f; //0.5f;	//the reduction in the points awarded for remaining pills when the level time runs out	
	
//	public static int PACMAN_SPEED_REDUCTION = 1;		// The frequency with which pacman moves: not implemented yet.
	public  int BOARD_COMPLETION_BONUS = 0; // 100	// Number of bonus points awarded for completing the board	
	
	//for Competition
	public static  int WAIT_LIMIT=5000;				//time limit in milliseconds for the controller to initialise;
	public static  int MEMORY_LIMIT=512;				//memory limit in MB for controllers (including the game)
	public static  int IO_LIMIT=10;					//limit in MB on the files written by controllers
	
	//for Maze
	public static int NUM_MAZES=7;					//number of different mazes in the game
	public static  String pathMazes="data/mazes";
	public static  String pathDistances="data/distances";
	public static  String[] nodeNames={"a","b","c","d", "mymaze","twoBoxConnect","fourBox"};
	public static  String[] distNames={"da","db","dc","dd", "dmymaze","dtwoBoxConnect","dfourBox"};
	public static String pathImages="data/images";
	public static String[] mazeNames={"maze-a.png","maze-b.png","maze-c.png","maze-d.png", "maze-d.png","maze-d.png","maze-d.png"};
	
	
	//for GameView
	public static  int MAG=2;
	public static  int GV_WIDTH=114;
	public static  int GV_HEIGHT=130;


	
	public Constants(){
		Random rng = new Random();
		SEED = rng.nextLong();
	}
	
	public void loadFromFile(String filename) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		String line;
		while ((line = reader.readLine()) != null){
			String[] keyValPair = line.split(":");
			if (keyValPair.length != 2){
				System.err.println("Error reading line: " + line);
				continue;
			}
			setVariable(keyValPair[0].trim(), keyValPair[1].trim());	
		}
		reader.close();
	}
	
	public void setLineGhostStartTimes(){
		GHOST.BLINKY.initialLairTime = 4;
		GHOST.INKY.initialLairTime = 6;
		GHOST.PINKY.initialLairTime = 8;
		GHOST.SUE.initialLairTime = 10;
	}
	
	public void setDefaultGhostStartTimes(){
		GHOST.BLINKY.initialLairTime = 40;
		GHOST.INKY.initialLairTime = 60;
		GHOST.PINKY.initialLairTime = 80;
		GHOST.SUE.initialLairTime = 100;
	}
	
	// For use with a file reader to populate constant values
	public void setVariable(String name, String value){
		if (name.equals("EDIBLE_TIME")){
			this.EDIBLE_TIME = Integer.parseInt(value);
		}
		else if (name.equals("COMMON_LAIR_TIME")){
			this.COMMON_LAIR_TIME = Integer.parseInt(value);
		}
		else if (name.equals("BOARD_COMPLETION_BONUS")){
			this.BOARD_COMPLETION_BONUS = Integer.parseInt(value);
		}
		else if (name.equals("LEVEL_LIMIT")){
			this.LEVEL_LIMIT = Integer.parseInt(value);
		}
		else if (name.equals("NUM_GHOSTS")){
			this.NUM_GHOSTS = Integer.parseInt(value);
		}
		else if (name.equals("GHOST_SPEED_REDUCTION")){
			this.GHOST_SPEED_REDUCTION = Integer.parseInt(value);
		}
		else if (name.equals("BLINKY")){
			GHOST.BLINKY.initialLairTime = Integer.parseInt(value);
		}
		else if (name.equals("INKY")){
			GHOST.INKY.initialLairTime = Integer.parseInt(value);
		}
		else if (name.equals("PINKY")){
			GHOST.PINKY.initialLairTime = Integer.parseInt(value);
		}
		else if (name.equals("SUE")){
			GHOST.SUE.initialLairTime = Integer.parseInt(value);
		}
		else if (name.equals("MAZE_NUM")){
			this.MAZE_NUM = Integer.parseInt(value);
		}
		else if (name.equals("SEED")){
			this.SEED = Long.parseLong(value);
		}
		else if (name.equals("GHOST_TYPE")){
			this.GHOST_TYPE = Integer.parseInt(value);
			if (this.GHOST_TYPE == 3){
				this.setLineGhostStartTimes();
			}
			else {
				this.setDefaultGhostStartTimes();
			}
		}
		else if (name.equals("SOURCE_TASK")){
			this.SOURCE_TASK = value;
		}
		else if (name.equals("LEARNER")){
			this.LEARNER = value;
		}
		else if (name.equals("LENGTH")){
			this.LENGTH = Integer.parseInt(value);
		}
		else if (name.equals("MAZE_NAME")){
			this.nodeNames[4] = value;
			this.distNames[4] = "d" + value;
		}
		else if (name.equals("SAVE_DIR")){
			this.SAVE_DIR = value;
		}
		else if (name.equals("PERFORMANCE_THRESHOLD")){
			this.PERFORMANCE_THRESHOLD = Double.parseDouble(value);
		}
		else {
			System.out.println("Constants ignored loading variable: " + name);
			return;
		}
		System.out.println("Loaded (" + name + ", " + value + ")");
	}
	
}