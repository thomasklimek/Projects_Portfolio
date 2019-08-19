

import mazeBuilder


def main():
  fourBox()

def fourBox():
  m = mazeBuilder.Maze()
  m.addRow(0,40)
  m.addCol(0,40)
  m.addCol(40,20)
  m.addRow(80,20)
  m.addNode(24,24)
  m.addRow(121,60)
  m.addCol(121,60)
  m.addRow(239,60)
  m.addCol(180,60)
  m.addNode(4,64)
  m.addRow(359,20)
  m.addCol(359,40)
  m.addRow(418,40)
  m.addCol(259,20)
  m.addNode(64,4)
  m.addCol(478,20)
  m.addRow(478,40)
  m.addCol(537,40)
  m.addRow(319,20)
  m.addRow(339,20)
  m.addCol(616,40)
  m.addCol(279,20)
  m.addRow(676,40)
  m.dispersePills(4)
  m.setPower(1)
  m.setPower(418)
  m.setPower(537)
  m.setPower(655)
  m.setGhostLair(50,50)
  m.setStartNodes(150,269)
  m.checkMaze()
  m.save('fourBox')

def twoBox():
  m = mazeBuilder.Maze()
  m.addRow(0, 100)
  m.addCol(0, 20)
  m.addRow(120, 100)
  m.addCol(100, 20)
  m.addCol(50, 20)
#  m.linkRow(14)
  m.dispersePills(4)
  m.setPower(250)
  m.setGhostLair(50,50)
  m.setStartNodes(0, 120)
  m.checkMaze()
  m.save('mymaze')

def twoBoxWithConnector():
  m = mazeBuilder.Maze()
  m.addRow(0,25)
  m.addCol(0,25)
  m.addRow(50,25)
  m.addCol(25,25)
  m.addRow(88,10)
  m.addNode(39,4)
  m.addCol(110,25)
  m.addRow(110,25)
  m.addRow(134,25)
  m.addCol(159,25)
  m.dispersePills(4)
  m.setPower(197)
  m.setGhostLair(50,55)
  m.setStartNodes(0,50)
  m.checkMaze()
  m.save('twoBoxConnect')





if __name__=="__main__":
  main()



