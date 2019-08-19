package pacman;

import pacman.game.Constants;


import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.Buffer;
import java.text.DecimalFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Queue;
import java.util.Random;

import javax.imageio.ImageIO;

import pacman.controllers.KeyBoardInput;
import pacman.entries.ghosts.*;
import pacman.entries.pacman.BasicRLPacMan;
import pacman.entries.pacman.CustomFeatureSet;
import pacman.entries.pacman.CustomFeatureSetV2;
import pacman.entries.pacman.DepthFeatureSet;
import pacman.entries.pacman.FeatureSet;
import pacman.entries.pacman.QPacMan;
import pacman.entries.pacman.RLPacMan;
import pacman.entries.pacman.SarsaPacMan;
import pacman.game.Game;
import pacman.game.GameView;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.internal.Maze;
import pacman.utils.DataFile;
import pacman.utils.LearningCurve;
import pacman.utils.Stats;

public class RunTest {
	
// Note on notation:
// custom refers to the CustomFeatureSet (using something else defaults to DepthFeatureSet)
// S refers to SarsaPacMan (using something else defaults to QPacMan)
	
	
/* 	4 options for RLPacMan learner type
 * 		- teacher: teacher RLPacMan loads policy
 * 		- independent: student starts from scratch
 * 		- transfer: student loads policy
 * 		- (baseline, advise, correct, predict): uses teacher+independent and a strategy for teacher 
 * 
 * 	Notes on folder names in mydata
 * 		- mydata/TEACHER/STUDENT/LEARNER-TYPE_MAZE-NUM/
 * 
 * 
 * 
 */	
	
	
	//options for features include depth or custom, options for algoirithms are S or Q
	//i.e., customQ, customS, depthS, depthQ are the legal values for STUDENT
	public static String STUDENT = "customS"; // Student feature set and algorithm
	
	
	public static int LENGTH = 500; // 10; // 100; // Points per curve
	public static int TEST = 1; //30; // Test episodes per point
	public static int TRAIN = 1; //10 // Train episodes per point

	public static Random rng = new Random();
	
	// Ghost "policies"
	public static StandardGhosts ghostsS = new StandardGhosts();
    public static RandomGhosts ghostsR = new RandomGhosts();
    public static ChaserGhosts ghostsC = new ChaserGhosts();
    public static LineGhosts ghostsL = new LineGhosts();
    
    public static int ghostType = 0;		// 0 = RandomGhosts, 1 = StandardGhosts, 2 = ChaserGhosts
    
    // Task parameters
    public static int mazeNum = 0;
    // Controls how fast ghosts move when pacman has eaten power pill. Lower numbers = slower (1 = frozen). 
    public static int ghostSlowdown = 2;
    public static int nGhosts = 4;
    
    public static Constants defaultConstants;
    
	/**
	 * Run experiments.
	 */
	public static void main(String[] args) {
		tamer();
		
	}
	
	public static void demo(){
		defaultConstants = new Constants();
		
		//train params
		LENGTH = 300; //how many train episodes
		TEST = 1;
				
		mazeNum = 2; //which maze;
		ghostType = 1; //which of three types of ghosts
		ghostSlowdown = 2; //ghost speed when edible
		nGhosts = 4; //how many ghosts
		
		defaultConstants.MAZE_NUM = mazeNum; //which maze;
		defaultConstants.GHOST_TYPE = ghostType; //which of three types of ghosts
		defaultConstants.GHOST_SPEED_REDUCTION = ghostSlowdown; //ghost speed
		defaultConstants.NUM_GHOSTS = nGhosts; //how many ghosts
		//defaultConstants.nodeNames[4]="mymaze";
		//defaultConstants.distNames[4]="dmymaze";

		//increase delay for watching
		defaultConstants.DELAY = defaultConstants.DELAY*5;
				
		//train
		String demo_name = "independent_temp";
			train(demo_name,0,false,false);
				
	}
	public static void compareRL(){
		defaultConstants = new Constants();
		
		//train params
		LENGTH = 300; //how many train episodes
		TEST = 1;
				
		mazeNum = 2; //which maze;
		ghostType = 1; //which of three types of ghosts
		ghostSlowdown = 2; //ghost speed when edible
		nGhosts = 4; //how many ghosts
		
		defaultConstants.MAZE_NUM = mazeNum; //which maze;
		defaultConstants.GHOST_TYPE = ghostType; //which of three types of ghosts
		defaultConstants.GHOST_SPEED_REDUCTION = ghostSlowdown; //ghost speed
		defaultConstants.NUM_GHOSTS = nGhosts; //how many ghosts
		//defaultConstants.nodeNames[4]="mymaze";
		//defaultConstants.distNames[4]="dmymaze";

		//increase delay for watching
		defaultConstants.DELAY = defaultConstants.DELAY*5;
				
		//train
		String demo_name = "independent_temp";
		for (int i = 0; i< 10; i++) {
			train(demo_name,0,false,false);
		}
				
	}
	public static void tamer(){
		defaultConstants = new Constants();
		
		//train params
		LENGTH = 300; //how many train episodes
		TEST = 1;
				
		mazeNum = 2; //which maze;
		ghostType = 1; //which of three types of ghosts
		ghostSlowdown = 2; //ghost speed when edible
		nGhosts = 4; //how many ghosts
		
		defaultConstants.MAZE_NUM = mazeNum; //which maze;
		defaultConstants.GHOST_TYPE = ghostType; //which of three types of ghosts
		defaultConstants.GHOST_SPEED_REDUCTION = ghostSlowdown; //ghost speed
		defaultConstants.NUM_GHOSTS = nGhosts; //how many ghosts
		//defaultConstants.nodeNames[4]="mymaze";
		//defaultConstants.distNames[4]="dmymaze";

		//increase delay for watching
		defaultConstants.DELAY = defaultConstants.DELAY*5;
				
		//train
		String demo_name = "independent_temp";
		//for (int i = 0; i< 9; i++) {
			tamerTrain(demo_name,0,false,false);
		//}
				
	}
	
	
	
	public static String generateDirectoryName(String learner, 
						int mazeNum_i, int ghostType_i, 
						int ghostSlowdown_i, int nGhosts){
		return new String(learner+"_maze"+mazeNum_i+"_type"+ghostType_i+"_slow"+ghostSlowdown_i+"_ghosts"+nGhosts+"");
	}

	/** Set up a learner. */
	public static RLPacMan create(String learner) {
		
		//create feature set based on constant STUDENT
		FeatureSet studentProto = STUDENT.startsWith("custom") ? new CustomFeatureSet() : new DepthFeatureSet();

		// Lone student
		if (learner.startsWith("independent")) {
			return STUDENT.endsWith("S") ? new SarsaPacMan(studentProto) : new QPacMan(studentProto);
		}
		
		return null;
	}
	/** Generate learning curves. */
	public static void tamerTrain(String learner, int start, boolean watchAtEnd, boolean watchDuring) {
		
		String directory = generateDirectoryName(learner,mazeNum,ghostType,ghostSlowdown,nGhosts);
		
		
		// This just sets up the student or teacher with the specified feature set and learning algorithm
		RLPacMan pacman = create(learner);
			
		// First point
		double[] initialData = pacman.episodeData();		// Starts off empty
			
		// evaluate the random policy
		double [] eval_result = evaluate(pacman, TEST);
		
		System.out.println("score at start = "+eval_result[0]);
			
		double initialScore = eval_result[0];		// This returns the average score over TEST number of games
		double initialTime = eval_result[1];		// This returns the average score over TEST number of games
			
		// Rest of the points
		int num_train_episodes = 0;
		
		// for each train episode...
		

		int trainTime = 0;
		while (trainTime < 200000) {
			double[] data = new double[initialData.length];
				
			
			//make PacMan play an episode
			//totalTime += episode(pacman);	
			if (num_train_episodes < 3)
				trainTime += tamerEpisode(pacman);
			else
				trainTime += episode(pacman);
			num_train_episodes++;
					
					
			double[] episodeData = pacman.episodeData();
			for (int d=0; d<data.length; d++)
				data[d] += episodeData[d];
			
			//System.out.println("timestep " + trainTime);
			double score = 0;
			for (int j = 0; j < 10; j++) {
				// now evaluate 	
				double [] eval_result2 = evaluate(pacman, TEST) ;
				score  += eval_result2[0];		// This returns the average score over TEST number of games
				
				//System.out.println(score);	
			}
			score = score / 5;
			System.out.println(trainTime+", "+score);
			
			
			if (watchDuring){
				((SarsaPacMan)pacman).debug=false;
				watch(pacman,true);
				((SarsaPacMan)pacman).debug=false;
			}
			
		}
		System.out.println("end tamer data");
		int totalTime = 0;
		pacman = create(learner);
		while (totalTime < 200000) {
			double[] data = new double[initialData.length];
				
			
			//make PacMan play an episode
			totalTime += episode(pacman);	

					
					
			double[] episodeData = pacman.episodeData();
			for (int d=0; d<data.length; d++)
				data[d] += episodeData[d];
			
			//System.out.println("timestep " + trainTime);
			double score = 0;
			for (int j = 0; j < 10; j++) {
				// now evaluate 	
				double [] eval_result2 = evaluate(pacman, TEST) ;
				score  += eval_result2[0];		// This returns the average score over TEST number of games
				
				//System.out.println(score);	
			}
			score = score / 10;
			System.out.println(totalTime+", "+score);
			
			
			
			if (watchDuring){
				((SarsaPacMan)pacman).debug=false;
				watch(pacman,true);
				((SarsaPacMan)pacman).debug=false;
			}
			
		}
		
		if (watchAtEnd)
			watch(pacman, false);
		
		
		System.out.println("Done.");
	}
	
	/** Generate learning curves. */
	public static void train(String learner, int start, boolean watchAtEnd, boolean watchDuring) {
		
		String directory = generateDirectoryName(learner,mazeNum,ghostType,ghostSlowdown,nGhosts);
		
		
		// This just sets up the student or teacher with the specified feature set and learning algorithm
		RLPacMan pacman = create(learner);
			
		// First point
		double[] initialData = pacman.episodeData();		// Starts off empty
			
		// evaluate the random policy
		double [] eval_result = evaluate(pacman, TEST);
		
		System.out.println("score at start = "+eval_result[0]);
			
		double initialScore = eval_result[0];		// This returns the average score over TEST number of games
		double initialTime = eval_result[1];		// This returns the average score over TEST number of games
			
		// Rest of the points
		int num_train_episodes = 0;
		
		// for each train episode...
		
		int totalTime = 0;
		while (totalTime < 1000000) {
		//for (int i = 0; i < 5; i++) {
			double[] data = new double[initialData.length];
				
			
			//make PacMan play an episode
			//totalTime += episode(pacman);	
			totalTime += episode(pacman);
			num_train_episodes++;
					
					
			double[] episodeData = pacman.episodeData();
			for (int d=0; d<data.length; d++)
				data[d] += episodeData[d];
			
				
				
			// now evaluate 	
			double [] eval_result2 = evaluate(pacman, TEST) ;
			double score = eval_result2[0];		// This returns the average score over TEST number of games
			double time = eval_result2[1];		// This returns the average number of game steps over TEST number of games
		
			System.out.println(totalTime+", "+score);
			
			//System.out.println(""+num_train_episodes+","+score);
			
			if (watchDuring){
				((SarsaPacMan)pacman).debug=false;
				watch(pacman,true);
				((SarsaPacMan)pacman).debug=false;
			}
			
		}
		
		if (watchAtEnd)
			watch(pacman, false);
		
		
		System.out.println("Done.");
	}
 
	/** Train a learner for one more episode. */
// Basically does the same thing as evaluate but doesn't return the score	
	public static int episode(RLPacMan pacman) {

		Game game = new Game(rng.nextLong(), defaultConstants);
		pacman.startEpisode(game, false);

		while(!game.gameOver()) {
			game.advanceGame(pacman.getMove(game.copy(), -1), getGhostMove(game, game.constants.GHOST_TYPE));				
			pacman.processStep(game);
		}
		return game.getTotalTime();
	}
	
	/** Estimate the current performance of a learner. */
// Width is the number of episodes to test for (i.e. variable TEST above)	
	public static double[] evaluate(RLPacMan pacman, int width) {
		
		double[] scoreAndTime = new double[2];		
		double sumScore = 0;
		double sumSteps = 0;

// For each test episode		
		for(int i=0; i<width; i++) {
// Create and start a new game			
			Game game = new Game(rng.nextLong(), defaultConstants);
			game.setTamer(false);
			pacman.startEpisode(game, true);
			while(!game.gameOver()) {
// getMove in pacman learner returns the lastAction	
// This will then recompute the next move to make.
				game.advanceGame(pacman.getMove(game.copy(), -1), getGhostMove(game, game.constants.GHOST_TYPE));
				//int key = KeyBoardInput.getKey();
				pacman.processStep(game);
			}
			
			sumScore += game.getScore();
			sumSteps += game.getTotalTime();
			//System.out.println("time taken: " + game.getTotalTime());
		}

		scoreAndTime[0] = sumScore/width;
		scoreAndTime[1] = sumSteps/width;
		
		
		return scoreAndTime;
	}

	
	
	public static void computeGraphFeatures(){
		Game game=new Game(rng.nextLong(), defaultConstants);
		
		//get maze
		Maze M = game.getCurrentMaze();
		
				
		//get number of nodes
		int num_nodes = M.graph.length;
			
		//get starting positions for ghosts and pacman
		int lair_index = M.lairNodeIndex;
		int ghost_start = M.initialGhostNodeIndex;
		int pacman_start = M.initialPacManNodeIndex;
		
		//get pill indeces
		int [] pill_indeces = M.pillIndices;
		int [] power_pill_indeces = M.powerPillIndices;
		
		//node to pill ratio:
		double node_to_pill_ratio = (double)num_nodes/(double)pill_indeces.length;
		
		//starting distance between pacman and ghosts
		int dist_ghost_to_pacman = game.getShortestPath(ghost_start, pacman_start).length;
		int dist_lair_to_pacman = game.getManhattanDistance(lair_index, pacman_start);
		
		//average distance between power pills
		double avg_pp_to_pp_distance = 0;
		int c = 0;
		for (int i = 0; i < power_pill_indeces.length; i ++){
			for (int j = 0; j < power_pill_indeces.length; j++){
				if (i!=j){
					avg_pp_to_pp_distance += (double)game.getShortestPath(ghost_start, pacman_start).length;
					c++;
				}
			}
		}
		avg_pp_to_pp_distance = avg_pp_to_pp_distance /(double)c;
		
		//average distance between power pills and lair
		double avg_lair_to_pp_distance = 0;
		for (int i = 0; i < power_pill_indeces.length; i ++){
			avg_lair_to_pp_distance += game.getShortestPath(ghost_start, power_pill_indeces[i]).length;
		}
		avg_lair_to_pp_distance = avg_lair_to_pp_distance / (double)power_pill_indeces.length;
		
		//average effective eccentricity
		double avg_effective_eccentricity = 0;
		double max_effective_eccentricity = Double.MIN_VALUE;
		for (int i = 0; i < M.graph.length; i ++){
			double max = Double.MIN_VALUE;
			for (int j = 0; j < M.graph.length; j ++){
			
				if (game.getNeighbouringNodes(M.graph[i].nodeIndex).length > 0
						&& game.getNeighbouringNodes(M.graph[j].nodeIndex).length > 0){
					double d_uv = game.getShortestPath(M.graph[i].nodeIndex, M.graph[j].nodeIndex).length;
					if (d_uv > max)
						max = d_uv;
				}
			}
			
			avg_effective_eccentricity+=max;
			
			if (max > max_effective_eccentricity)
				max_effective_eccentricity = max;
		}
		avg_effective_eccentricity = avg_effective_eccentricity / (double)M.graph.length;
		
		//average effective eccentricity for between junctions
		double avg_effective_eccentricity_junct = 0;
		double max_effective_eccentricity_junct = Double.MIN_VALUE;
		for (int i = 0; i < M.junctionIndices.length; i ++){
			double max = Double.MIN_VALUE;
			for (int j = 0; j < M.junctionIndices.length; j ++){
			
				if (game.getNeighbouringNodes(M.junctionIndices[i]).length > 0
						&& game.getNeighbouringNodes(M.junctionIndices[j]).length > 0){
					double d_uv = game.getShortestPath(M.junctionIndices[i], M.junctionIndices[j]).length;
					if (d_uv > max)
						max = d_uv;
				}
			}
			
			avg_effective_eccentricity_junct+=max;
			
			if (max > max_effective_eccentricity_junct)
				max_effective_eccentricity_junct = max;
		}
		avg_effective_eccentricity_junct = avg_effective_eccentricity_junct / (double)M.junctionIndices.length;
		
		//for each pair of junctions, compute shortest path and count how many junctions are on it
		double avg_num_junctions_between_junctions = 0;
		for (int i = 0; i < M.junctionIndices.length; i ++){
			for (int j = 0; j < M.junctionIndices.length; j ++){
				int [] path_ij = game.getShortestPath(M.junctionIndices[i], M.junctionIndices[j]);
				
				double num_juncts_in_path = 0;
				
				for (int k = 0; k < path_ij.length; k++){
					for (int p = 0; p < M.junctionIndices.length; p ++){
						if (path_ij[k]==M.junctionIndices[p] && i != p && j != p)
							num_juncts_in_path+=1.0;
					}
				}
				
				avg_num_junctions_between_junctions += num_juncts_in_path;
			}
		}
		avg_num_junctions_between_junctions = avg_num_junctions_between_junctions/(M.junctionIndices.length*M.junctionIndices.length);
		
		//histogram of degrees of nodes
		int [] hist = new int[5];
		for (int i = 0; i < 5; i ++){
			hist[i]=0;
		}
		for (int i = 0; i < M.graph.length; i ++){
			int degree_i = game.getNeighbouringNodes(M.graph[i].nodeIndex).length;
			hist[degree_i]++;
		}
		
		
		
		System.out.println("num_nodes\t"+num_nodes);
		//System.out.println("starting positions:\t"+lair_index+"\t"+ghost_start+"\t"+pacman_start);
		System.out.println("num_pills\t"+pill_indeces.length);
		//and power pills:\t"+pill_indeces.length+"\t"+power_pill_indeces.length);
		
		System.out.println("node_to_pill_ratio\t"+node_to_pill_ratio);
		System.out.println("dist_ghost_to_pacman\t"+dist_ghost_to_pacman);
	
		System.out.println("avg_pp_to_pp_distance\t"+avg_pp_to_pp_distance);
		System.out.println("avg_lair_to_pp_distance\t"+avg_lair_to_pp_distance);
		System.out.println("avg_effective_eccentricity\t"+avg_effective_eccentricity);
		System.out.println("avg_effective_eccentricity_junct\t"+avg_effective_eccentricity_junct);
		
		System.out.println("max_effective_eccentricity\t"+max_effective_eccentricity);
		System.out.println("max_effective_eccentricity_junct\t"+max_effective_eccentricity_junct);
		
		System.out.println("avg_num_junctions_between_junctions\t"+avg_num_junctions_between_junctions);
		
		//System.out.println("Histogram of degrees:");
		for (int i = 2; i < 5; i ++){
			System.out.println( "dhist_"+(i)+"\t"+hist[i]);
		}
		
		
	}
	
	/** Observe a learner play a game. */
	public static void watchAndSave(RLPacMan pacman, boolean destroyWindow, String prefix, String path) {
		
		
		Game game=new Game(rng.nextLong(), defaultConstants);
		
		pacman.startEpisode(game, true);
		GameView gv=new GameView(game).showGame();
		int c = 0;
		

		while(!game.gameOver()) {
			
			game.advanceGame(pacman.getMove(game.copy(), -1), getGhostMove(game, game.constants.GHOST_TYPE));	
			pacman.processStep(game);
			
			try{Thread.sleep(defaultConstants.DELAY);}catch(Exception e){}
			gv.repaint();
	
			//Image img_i = gv.createImage(100, 100);
			Image img_i = gv.getImage();
			BufferedImage bi = (BufferedImage)img_i;
			
			
			StringBuffer leadingZeros = new StringBuffer();
			if (c < 10)
				leadingZeros.append("0000");
			else if (c < 100)
				leadingZeros.append("000");
			else if (c < 1000)
				leadingZeros.append("00");
			else if (c < 10000)
				leadingZeros.append("0");
			
			
			
			File f = new File(new String(path+"/img"+leadingZeros.toString()+c+".png"));
			c++;
			try {
				ImageIO.write(bi, "png", f);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		if (destroyWindow){
			gv.setVisible(false);
			gv.setEnabled(false);
			gv.getFrame().dispose();
		}
	}

	/** Observe a learner play a game. */
	public static void watch(RLPacMan pacman, boolean destroyWindow) {
		
		
		Game game=new Game(rng.nextLong(), defaultConstants);
		
		pacman.startEpisode(game, true);
		GameView gv=new GameView(game).showGame();
		
		

		while(!game.gameOver()) {
			
			game.advanceGame(pacman.getMove(game.copy(), -1), getGhostMove(game, game.constants.GHOST_TYPE));	
			pacman.processStep(game);
			
			try{Thread.sleep(defaultConstants.DELAY);}catch(Exception e){}
			gv.repaint();
			Image img_i = gv.createImage(100, 100);
			BufferedImage bi = (BufferedImage)img_i;
			File f = new File("./output.png");
			try {
				ImageIO.write(bi, "png", f);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		if (destroyWindow){
			gv.setVisible(false);
			gv.setEnabled(false);
			gv.getFrame().dispose();
		}
	}
	public static int tamerEpisode(RLPacMan pacman) {
		KeyBoardInput l = new KeyBoardInput();
		Game game=new Game(rng.nextLong(), defaultConstants);
		game.setTamer(true);
		
		pacman.startEpisode(game, false);
		GameView gv=new GameView(game).showGame();
		gv.getFrame().addKeyListener(l);
		
		int score = 0;
		while(!game.gameOver()) {
			
				
			
			
			try{Thread.sleep(defaultConstants.DELAY * 3);}catch(Exception e){}
			gv.repaint();
			
			int key = l.getKey();
			//System.out.println(key);
			if (key == 49) {
				score += 5;
			}
			if(key == 50) {
				score -= 5;
			}
			
			game.advanceGame(pacman.getMove(game.copy(), -1), getGhostMove(game, game.constants.GHOST_TYPE));
			game.setScore(score);
			pacman.processStep(game);
		}
		return game.getTotalTime();
	}
	
	
	private static EnumMap<GHOST, MOVE> getGhostMove(Game game, int ghostType){
		
		switch(ghostType){
			case 0:
				return ghostsR.getMove(game.copy(), -1);
			case 1:
				return ghostsS.getMove(game.copy(), -1);
			case 2:
				return ghostsC.getMove(game.copy(), -1);
			case 3:
				return ghostsL.getMove(game.copy(), -1);
			default:
				System.err.println("INVALID GHOST TYPE");
				System.exit(1);
				return null;				// Compilier is complaining... 
		}
	}
}