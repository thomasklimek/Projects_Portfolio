package pacman.entries.ghosts;

import java.util.EnumMap;
import java.util.Random;

import pacman.controllers.Controller;
import pacman.game.Constants.DM;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Game;

/**
 * One ghost uses default behavior. Each subsequent ghost follows the one before it in a line. 
 */
public class LineGhosts extends Controller<EnumMap<GHOST,MOVE>>
{
	private Random rng = new Random();
	private static final double CONSISTENCY = 0.8;
	
	private EnumMap<GHOST,MOVE> myMoves = new EnumMap<GHOST,MOVE>(GHOST.class);

	/** Fill all ghost moves. */
	public EnumMap<GHOST, MOVE> getMove(Game game, long timeDue) {
				
		myMoves.clear();

		for (GHOST ghost : GHOST.values())
			if (game.doesGhostRequireAction(ghost))
				myMoves.put(ghost,getMove(ghost, game));

		return myMoves;
	}
	
	/** Fill one ghost move. */
	private MOVE getMove(GHOST ghost, Game game) {
		
		DM metric = DM.PATH;
		MOVE[] allMoves = MOVE.values();
		
		MOVE myLastMove = game.getGhostLastMoveMade(ghost);
						
		int sourceNode = game.getGhostCurrentNodeIndex(ghost);		
		int targetNode = game.getPacmanCurrentNodeIndex();
		int pacmanNode = game.getPacmanCurrentNodeIndex();
		
		
		// Sort the ghosts by closest to pacman
		double[] distances = new double[GHOST.values().length];
		int index = 0;
		double myDistance = game.getDistance(sourceNode, pacmanNode, myLastMove, metric);
		for (GHOST g: GHOST.values()){
//			if (game.getGhostEdibleTime(g) > 0)
//				distances[index++] = Double.POSITIVE_INFINITY;
//			else
				distances[index++] = game.getDistance(game.getGhostCurrentNodeIndex(g), pacmanNode, game.getGhostLastMoveMade(g), metric);
		}
		
		// If anyone is closer to pacman than you, follow them
		for (int i = 0; i < distances.length; i++){
			if (distances[i] < myDistance){
				myDistance = distances[i];
				targetNode = game.getGhostCurrentNodeIndex(GHOST.values()[i]);
			}
		}
					
/*		
		for (GHOST previousGhost: GHOST.values()){			
			if (previousGhost.compareTo(ghost) < 0 && myMoves.get(previousGhost) != null){
				targetNode = game.getGhostCurrentNodeIndex(previousGhost);
			}		
		}
*/		
						
		MOVE myNextMove = game.getApproximateNextMoveTowardsTarget(sourceNode,targetNode,myLastMove,metric);		
		
		// When edible, just run away!!
		if (game.getGhostEdibleTime(ghost) > 0)
			myNextMove = game.getApproximateNextMoveAwayFromTarget(sourceNode,pacmanNode,myLastMove,metric);

		
		return myNextMove;
	}
}
