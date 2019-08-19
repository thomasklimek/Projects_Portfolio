package pacman.entries.pacman;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import pacman.game.Constants;
import pacman.game.Constants.DM;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Game;
import pacman.game.GameView;
import pacman.game.IO;
import pacman.game.internal.Ghost;

/**
 * Specially crafted features that work well in this domain.
 */
public class CustomFeatureSetV2 extends FeatureSet {
	
	private int DEPTH = 4; // Of search 	// 4
	
	// DEPTH only controls how many branches/junctions to search upto. Otherwise, it will keep following the path or corner until a junction is reached.
	
	
	private int FEATURES = 7; // How many (DEPTH + 3)
	private double MAX_DISTANCE = 200; // Between nodes
	private double MAX_SCORE = Math.pow(MAX_DISTANCE/4,2); // Of a path

	// Computation storage
// There will be DEPTH hashmaps in this arraylist
// For each depth level (i.e. junction), you populate that junction node with its (max) safety score. 	
	
	private ArrayList<HashMap<Integer,Double>> junctions;
	
	
	
	
	private double powerDepth = DEPTH;
	private double pillDepth = DEPTH;
	private boolean pillInPowerSeg = false;
	

	// Feature values
	public double[] values = new double[FEATURES];

	/** Set up data structures. */
	public CustomFeatureSetV2() {
		junctions = new ArrayList<HashMap<Integer,Double>>();
		for (int i=0; i<DEPTH; i++)
			junctions.add(new HashMap<Integer,Double>());
	}

	/** Report how many features there are. */
	public int size() {
		return values.length;
	}

	/** Retrieve a feature value. */
	public double get(int i) {
		return values[i];
	}

	/** Extract a feature set for this state-action pair. */
// So this is similar to the action-dependent features in HFO	
	public FeatureSet extract(Game game, MOVE move) {
		CustomFeatureSetV2 features = new CustomFeatureSetV2();
		features.setValues(game, move);
		return features;
	}

	/** Compute feature values in [0,1). */
	private void setValues(Game game, MOVE move) {
		
		int node = game.getPacmanCurrentNodeIndex();
		int next_node = game.getNeighbour(node, move);
		ArrayList<Ghost> game_ghosts = game.getGhosts();
		
		
		//global features
		
		//1. Pills eaten ratio - I don't think this helps
		double pills_eaten_fraction = (double)game.getNumberOfActivePills() / (double)game.getNumberOfPills();
	
		//3. shortest distance to a pill in a certain direction - returns MAX is there is a ghost on the path to that pill
		
		
		int [] pill_indices = game.getPillIndices(); //this returns all pills including the ones that were eaten
		int [] active_pill_indices = game.getActivePillsIndices();
		
		int shortest_distance = Integer.MAX_VALUE;
		int [] path_to_closest_pill = new int[1];
		for (int r = 0; r < active_pill_indices.length; r++){
			if ( active_pill_indices[r] != node ){
				int [] temp_path_to_closest_pill = game.getAStarPath(next_node, active_pill_indices[r], move);
				int d_r = temp_path_to_closest_pill.length;
				if (d_r < shortest_distance){
					shortest_distance = d_r;
					path_to_closest_pill = temp_path_to_closest_pill;
				}
			}
		}
		
		
		//check if a ghost lies on the path
		for (int r = 0; r < path_to_closest_pill.length; r++){
			for (Ghost g : game_ghosts){
				if ( g.edibleTime ==0 && g.currentNodeIndex == path_to_closest_pill[r]){
					shortest_distance = (int)MAX_DISTANCE;
				}
			}
		}
		
		
		
		
		double pill_distance_ratio = (MAX_DISTANCE-(double)shortest_distance)/((double)MAX_DISTANCE);
		//if (path_to_closest_pill.length > 1)
		//	GameView.addLines(game, new Color(0,255,255), next_node, path_to_closest_pill[path_to_closest_pill.length-1]);
		
		
		
		//4. Ghost input
		//b(c) = distance between the nearest non scared ghostand and the nearest intersection in that direction
		//d(c) distance to the nearest interesection in the direction
		
		//System.out.println("move: "+move.name()+"\t"+shortest_distance);
		
		//find closest junction in direction
		/*shortest_distance = Integer.MAX_VALUE;
		int closest_junction = -1;
		int [] junctions_maze = game.getJunctionIndices();
		for (int i = 0; i < junctions_maze.length; i++){
			int d_r = game.getAStarPath(next_node, junctions_maze[i], move).length;
			if (d_r < shortest_distance){
				shortest_distance = d_r;
				closest_junction = junctions_maze[i];
			}
		}
		
		GameView.addLines(game, new Color(0,255,255), next_node, closest_junction);
		
		//find the closest ghost to that junction
		shortest_distance = (int)MAX_DISTANCE;
		int closest_ghost_to_junction = -1;
		for (Ghost g : game_ghosts){
			if (g.edibleTime == 0 && g.lairTime == 0){
				int d_r = game.getManhattanDistance(closest_junction, g.currentNodeIndex);
				if (d_r < shortest_distance){
					shortest_distance = d_r;
					closest_ghost_to_junction = g.currentNodeIndex;
				}
			}
		}
		
		if (closest_ghost_to_junction != -1)
			GameView.addLines(game, new Color(255,0,255), closest_junction, closest_ghost_to_junction);
		
		
		int b_c = shortest_distance;
		int d_c = game.getAStarPath(next_node, closest_junction, move).length;
		
		//System.out.println(b_c+"\t"+d_c);
		double ghost_input = -1*(MAX_DISTANCE+d_c-b_c)/MAX_DISTANCE;
		*/
		
		//4.5 Closest ghost in the direction
		
		
		//5. scared ghost input
		int shortest_distance_to_scared_ghost = (int)MAX_DISTANCE;
		int shortest_distance_to_ghost = (int)MAX_DISTANCE;
		for (Ghost g : game_ghosts){
			if ( g.edibleTime > 0 && g.lairTime == 0){
				//System.out.println(next_node+"\t"+g.currentNodeIndex);
				int d_r = game.getAStarPath(next_node, g.currentNodeIndex, move).length;
				if (d_r < shortest_distance_to_scared_ghost){
					shortest_distance_to_scared_ghost = d_r;
				}
			}
			else if (g.lairTime == 0) {
				int d_r = game.getAStarPath(next_node, g.currentNodeIndex, move).length;
				if (d_r < shortest_distance_to_ghost){
					shortest_distance_to_ghost = d_r;
				}
			}
		}
		
		double ghost_afraid_input = (MAX_DISTANCE - (double)shortest_distance_to_scared_ghost)/MAX_DISTANCE;
		double ghost_input = (MAX_DISTANCE - (double)shortest_distance_to_ghost)/MAX_DISTANCE;
		
		//game_ghosts.clear();
		
		//TO DO: entrampment input
		
		
		//System.out.println("closeb_c+"\t"+d_c);
		
		// Things ahead?
		
		exploreJunctions(game, node, move, 0, 0);

		// Safety of junctions?
// Basically checks the most safe junction at each depth, and constrains by first junction.
// Seems like it should be constraining by all the ones before and not just the first... 
		double[] safety = new double[DEPTH];
		for (int i=0; i<DEPTH; i++) {
			for (Integer n : junctions.get(i).keySet()) {
				safety[i] = Math.max(safety[i], junctions.get(i).get(n));
				safety[i] = Math.min(safety[0], safety[i]);
			}
		}

		// Feast opportunity?
		double[] startPath = {MAX_DISTANCE, MAX_DISTANCE, MAX_DISTANCE, MAX_DISTANCE};
		double[] feastPath = exploreFeasts(game, node, move, 0, 0, startPath);
		double feastScore = score(feastPath);

		// Upcoming feast opportunity?
		double futureFeastScore = 0;
		MOVE[] possibleMoves = game.getPossibleMoves(node);
		for (MOVE possibleMove : possibleMoves) {
			double[] initialPath = {MAX_DISTANCE, MAX_DISTANCE, MAX_DISTANCE, MAX_DISTANCE};
			double[] path = exploreEnemies(game, node, possibleMove, 0, 0, initialPath);
			futureFeastScore = Math.max(futureFeastScore, score(path));
		}

		// Features
		int v = 0;
		for (int i=0; i<DEPTH; i++)
			values[v++] = safety[i] / MAX_DISTANCE; // How safe junctions are
		
		//original
		values[v++] = (DEPTH - pillDepth) / DEPTH; // How close a pill is
		
		//modified
		/*double pill_feature = (DEPTH - pillDepth) / DEPTH;
		if (pill_feature == 0.0 && values[0] > 0.0 ){
			if ( powerDepth != 0)
				pill_feature = pill_distance_ratio*(1.0/(double)DEPTH);
			else 
				pill_feature = 0.01;
				//System.out.println("ignoring pills close to power pill");
		}
		values[v++] = pill_feature;*/
				
		//values[v++] = (DEPTH - powerDepth) / DEPTH; // How close a pill is
		values[v++] = feastScore / MAX_SCORE; // How good feasts look
		values[v++] = futureFeastScore / MAX_SCORE / (powerDepth+1); // How good future feasts look*/
	
		// New features
		//values[v++] = (game.getTotalTime()  * 1.0) / Constants.LEVEL_LIMIT;
		
		
		//values[v++] = pills_eaten_fraction;
		//values[v++] = avg_edilbe_time/(double)Constants.EDIBLE_TIME;
		
		values[v++] = pill_distance_ratio;
		values[v++] = ghost_input;
		values[v++] = ghost_afraid_input;
		
		//System.out.println("Features for move:\t"+move);
		//IO.printDoubleArray(values);
		
		for (int k = 0; k < values.length; k++){
			if (Double.isNaN(values[k])){
				System.out.println("Found NaN:");
				System.out.println(values[0]+"\t"+values[1]+"\t"+values[2]);
			}
		}
		
		if (v != values.length) {
			System.out.println("Feature vector length error: said "+values.length+", got "+v);
			System.exit(0);
		}
	}

	/** Search up to a few junctions ahead in this direction. */
//	Searches up to DEPTH junctions, calculates the safety of that node, and populates it into junctions
	private void exploreJunctions(Game game, int node, MOVE move, int depth, double distance) {

		// This is incremented below every time you see a junction
		if (depth >= DEPTH)
			return;

		boolean pillInSegment = false;
		boolean powerInSegment = false;

		// Step
		while (true) {
			node = game.getNeighbour(node, move);
			distance++;

// Basically nothing gets populated if there is a ghost on the next node			
			// Stop for an approaching enemy
			for (GHOST ghost : GHOST.values()) {
				if (game.getGhostCurrentNodeIndex(ghost) == node)
					if (move != game.getGhostLastMoveMade(ghost))
						if (!game.isGhostEdible(ghost))
							return;
			}

			// Notice a power pill
// This part updates powerDepth			
			int powerIndex = game.getPowerPillIndex(node);
			if (powerIndex > -1 && game.isPowerPillStillAvailable(powerIndex)) {

				double safety = safety(game, node, distance);
				if (safety <= 0)
					return;

				powerInSegment = true;
				if (depth < powerDepth){
					powerDepth = depth;
					
				}
			}

			// Notice a regular pill (maybe)
			int pillIndex = game.getPillIndex(node);
			if (pillIndex > -1 && game.isPillStillAvailable(pillIndex))
				pillInSegment = true;

			// Notice a junction
			if (game.isJunction(node)) {
				
				double safety = safety(game, node, distance);
				if (safety <= 0)
					return;

				if (!junctions.get(depth).containsKey(node) || junctions.get(depth).get(node) < safety)
					junctions.get(depth).put(node, safety);

				// Really notice a regular pill (only in segments without power and with safe exits)
				if (!powerInSegment && pillInSegment && depth < pillDepth){
					pillDepth = depth;
					pillInPowerSeg = false;
				}
				else if (pillInSegment && depth < pillDepth){
					pillInPowerSeg = true;
				}
			}
			

			// Split at a junction
			if (game.isJunction(node)) {

				MOVE[] possibleMoves = game.getPossibleMoves(node, move);
				for (MOVE possibleMove : possibleMoves) {
					exploreJunctions(game, node, possibleMove, depth+1, distance);
				}

				return;
			}

			// Turn at a corner
			else if (game.getNeighbour(node, move) == -1)
				move = game.getPossibleMoves(node, move)[0];
		}
	}

	/** Find the highest-scoring path (ghost : depth) towards enemies in this direction. */
	private double[] exploreEnemies(Game game, int node, MOVE move, int depth, double distance, double[] path) {

		if (depth >= DEPTH-1)
			return path;

		// Step
		while (true) {
			node = game.getNeighbour(node, move);
			distance++;

			// Notice an approaching enemy
			for (GHOST ghost : GHOST.values()) {
				if (game.getGhostCurrentNodeIndex(ghost) == node)
					if (move != game.getGhostLastMoveMade(ghost))
						if (!game.isGhostEdible(ghost))
							if (distance < path[ghost.ordinal()])
								path[ghost.ordinal()] = distance;
			}

			// Split at a junction
			if (game.isJunction(node)) {

				double[] bestPath = path;
				double bestScore = score(path);

				MOVE[] possibleMoves = game.getPossibleMoves(node, move);
				for (MOVE possibleMove : possibleMoves) {

					double[] pathCopy = Arrays.copyOf(path, path.length);
					double[] newPath = exploreEnemies(game, node, possibleMove, depth+1, distance, pathCopy);
					double newScore = score(newPath);

					if (newScore > bestScore) {
						bestPath = newPath;
						bestScore = newScore;
					}
				}

				return bestPath;
			}

			// Turn at a corner
			else if (game.getNeighbour(node, move) == -1)
				move = game.getPossibleMoves(node, move)[0];
		}
	}

	/** Find the path with shortest distances towards feasts in this direction. */
// Path contains for each ghost, the minimum distance of a safe node	
	private double[] exploreFeasts(Game game, int node, MOVE move, int depth, double distance, double[] path) {

		if (depth >= DEPTH+1)
			return path;

		// Step
		while (true) {
			node = game.getNeighbour(node, move);
			distance++;

			// Stop for a power pill
			int powerIndex = game.getPowerPillIndex(node);
			if (powerIndex > -1 && game.isPowerPillStillAvailable(powerIndex))
				return path;

			for (GHOST ghost : GHOST.values()) {
				if (game.getGhostCurrentNodeIndex(ghost) == node) {

					// Stop for an approaching enemy
					if (!game.isGhostEdible(ghost)) {
						if (move != game.getGhostLastMoveMade(ghost))
							return path;
					}

					else {
						
						// Stop for an unsafe feast
						double safety = safety(game, node, distance);
						if (safety <= 0)
							return path;

						// Otherwise notice it
						if (distance < path[ghost.ordinal()])
							path[ghost.ordinal()] = distance;
					}
				}
			}

			// Split at a junction
			if (game.isJunction(node)) {

				double[] bestPath = path;
				double bestScore = score(path);

				MOVE[] possibleMoves = game.getPossibleMoves(node, move);
				for (MOVE possibleMove : possibleMoves) {

					double[] pathCopy = Arrays.copyOf(path, path.length);
					double[] newPath = exploreFeasts(game, node, possibleMove, depth+1, distance, pathCopy);
					double newScore = score(newPath);

					if (newScore > bestScore) {
						bestPath = newPath;
						bestScore = newScore;
					}
				}

				return bestPath;
			}

			// Turn at a corner
			else if (game.getNeighbour(node, move) == -1)
				move = game.getPossibleMoves(node, move)[0];
		}
	}

	/** Compute the score of a ghost path. Closer ghosts mean higher scores. */
	private double score(double[] path) {
		double score = 0;
		for (int i=0; i<path.length; i++)	
			score += Math.pow((MAX_DISTANCE - path[i])/4, 2);
		return score / path.length;
	}

	/** Compute the safety of a target node. Further ghosts means safer nodes. */
// Calculates a score of safety for reaching a node that is myDistance away from current position
// It's basically just the difference in distance between you and the closest ghost to the target. 	
	private double safety(Game game, Integer node, double myDistance) {
		double[] enemyDistances = enemyNodeDistances(game, node);
		Arrays.sort(enemyDistances);
		/*for (int i = 0; i < enemyDistances.length; i++){
			System.out.print(enemyDistances[i]+"\t");
		}
		System.out.println();*/
		if (enemyDistances[0] == MAX_DISTANCE)
			return enemyDistances[0];// - myDistance - game.constants.EAT_DISTANCE;
		else 
			return enemyDistances[0] - myDistance - game.constants.EAT_DISTANCE;
	}

	/** Compute relevant enemy distances to a nearby node. */
// If the ghost is edible, he is shown as max distance. This is paired with safety, so essentially
// if a ghost is edible, that node will have a high safety score.
	private double[] enemyNodeDistances(Game game, int node) {

		double[] distances = new double[GHOST.values().length];
		for (int i=0; i<distances.length; i++)
			distances[i] = MAX_DISTANCE;

		for (GHOST ghost : GHOST.values()) {
			if (!game.isGhostEdible(ghost)) {
				int ghostNode = game.getGhostCurrentNodeIndex(ghost);

				// Ignore ghosts in the lair
				if (ghostNode != game.getCurrentMaze().lairNodeIndex) {

					// Ignore ghosts I could reach before the target (close followers)
					int myNode = game.getPacmanCurrentNodeIndex();
					if (game.getDistance(myNode, node, DM.PATH) < game.getDistance(myNode, ghostNode, DM.PATH)) {

						MOVE ghostMove = game.getGhostLastMoveMade(ghost);
						distances[ghost.ordinal()] = game.getDistance(ghostNode, node, ghostMove, DM.PATH);
					}
				}
			}
		}

		return distances;
	}
}
