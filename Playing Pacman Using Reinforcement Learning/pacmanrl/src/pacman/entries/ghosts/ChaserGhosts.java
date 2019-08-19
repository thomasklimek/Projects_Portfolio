package pacman.entries.ghosts;

import java.util.EnumMap;
import java.util.Random;

import pacman.controllers.Controller;
import pacman.game.Constants.DM;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Game;

/**
 * Always chase when edible
 * Chase with some probability of moving randomly when not edible
 */
public class ChaserGhosts extends Controller<EnumMap<GHOST,MOVE>>
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
		
		//MOVE myNextMove = game.getNextMoveTowardsTarget(sourceNode, targetNode, myLastMove, metric);
		
		MOVE myNextMove = game.getApproximateNextMoveTowardsTarget(sourceNode,targetNode,myLastMove,metric);

		
		// Move randomly only when not edible *sometimes*
		if (game.getGhostEdibleTime(ghost) <= 0 && rng.nextDouble() > CONSISTENCY){
			myNextMove = allMoves[rng.nextInt(allMoves.length)];	// Random move
		}

		
		return myNextMove;
	}
}
