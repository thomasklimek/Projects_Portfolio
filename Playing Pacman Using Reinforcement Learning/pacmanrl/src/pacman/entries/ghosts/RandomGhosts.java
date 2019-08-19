package pacman.entries.ghosts;

import java.util.EnumMap;
import java.util.Random;

import pacman.controllers.Controller;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Game;

public final class RandomGhosts extends Controller<EnumMap<GHOST,MOVE>>
{
    private EnumMap<GHOST,MOVE> moves=new EnumMap<GHOST,MOVE>(GHOST.class);
    private MOVE[] allMoves=MOVE.values();
    private Random rnd=new Random();

    public EnumMap<GHOST,MOVE> getMove(Game game,long timeDue)
    {
        moves.clear();

        for(GHOST ghostType : GHOST.values())
            moves.put(ghostType,allMoves[rnd.nextInt(allMoves.length)]);

        return moves;
    }
}