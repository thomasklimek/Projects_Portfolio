
from Agent import Agent
from Environment import Environment
import MusicBank as mb
import mido
import pygame.mixer as pgm
import pygame.time as pgt
import numpy as np
import keyboard
import argparse
import sys

TAMER_TRIALS = 33

# ARGS
parser = argparse.ArgumentParser(description='RL Agent Harmonizer')
parser.add_argument('-m', help='Learn a melody from a midi file')
parser.add_argument('-p', help='Play midi file')
parser.add_argument('-d', help='Debugging', action='store_true')
parser.add_argument('-c', help='Harmony midi file. If this and ci flag not set, TAMER assumed')
parser.add_argument('-ci', help='Indicate chords includes in melody midi', action='store_true')
parser.add_argument('-s', help='Save the output', action='store_true')
parser.add_argument('-sp', help='Save and playback the output', action='store_true')
parser.add_argument('-t', help='Specify number of training trials, default = 10', type=int, default=10)
parser.add_argument('-a', default='A', help='String containing letters: M=major chords, m=minor, d=diminished, a=augmented, A=all. E.g. Mma')
parser.add_argument('-e1', help='run experiment1 [test file]', type=str)
parser.add_argument('-e2', help='run experiment2 [train file]', nargs=1, type=str)
parser.add_argument('-k', help='key of song for evaluation in form root+type, e.g. cmajor, g#minor', nargs=1, type=str)
args = parser.parse_args()

def main():
	# just play a midi file
	if args.p:
		play(args.p, args.d)
		return

	if args.e1:
		if not args.k:
			print("Must specify corresponding key [-k [key]]")
			return
		totalScores = []
		for i in range(10):
			args.m = None
			midi = setup_midi(args)
			agent = train(args, midi, setup_agent(args, midi))
			weights = agent.w
			print("-------testing with " + args.e1 + '-------')
			args.m = args.e1
			midi = setup_midi(args)
			agent = setup_agent(args, midi)
			agent.w = weights
			print("weights:", weights)
			fas,scores = evaluate(agent, midi, args.k[0])
			totalScores.append(scores)
		print("AVERAGES")
		print("correct\tfunction\tin key")
		ts = [float(sum(score))/len(score) for score in zip(*totalScores)]
		print("{0:1.4}\t".format(ts[0]) + "{0:1.4}\t".format(ts[1]) +
		"{0:1.4}".format(ts[2]))
		if args.s or args.sp:
			generate_output(agent, midi, args.s, args.sp, '-e2.mid', fas)

	elif args.e2:
		if not (args.c or args.ci):
			print("Must specify corresponding chords [-c [file] or -ci]")
			return
		if not args.k:
			print("Must specify corresponding key [-k [key]]")
			return
		args.m = args.e2[0]
		midi = setup_midi(args)
		print("TAMING with " + args.e2[0])
		temp = args.c
		args.c = args.ci = False
		midi2 = setup_midi(args)
		agent = train(args, midi2, setup_agent(args, midi2))
		args.ci = True
		args.c = temp
		midi2['chords'] = midi['chords']
		print("----DONE TAMING-----")
		fas,scores = evaluate(agent, midi2, args.k[0])
		if args.s or args.sp:
			generate_output(agent, midi, args.s, args.sp, '-e2.mid', fas)
	else:
		midi = setup_midi(args)
		agent = train(args, midi, setup_agent(args,midi))
		generate_output(agent, midi, args.s, args.sp, '-output.mid')


def setup_midi(Args):
	midi = {}
	if Args.m:
		# process input midi
		if Args.c:
			midi = process_midi_input(Args.m, Args.c)
		else:
			midi = process_midi_input(Args.m)
	else:
		# use preset training music
		Args.m = 'training.mid'
		midi = pretrain_midi()

	# midi defaults
	if 'tempo' not in midi:
		midi['tempo'] = 500000
	if 'measures' not in midi:
		midi['measures'] = len(midi['states'])
	if 'tpm' not in midi:
		midi['tpm'] = 480*4
	if 'cpc' not in midi:
		midi['cpc'] = 24
	if 'numerator' not in midi:
		midi['numerator'] = 4
	if 'denominator' not in midi:
		midi['denominator'] = 4

	# set actions from CL input
	midi['actions'] = []
	if 'A' in Args.a:
		midi['actions'] = mb.ALL_CHORDS
	else: 
		if 'M' in Args.a:
			midi['actions'] += mb.MAJOR_CHORDS
		if 'm' in Args.a:
			midi['actions'] += mb.MINOR_CHORDS
		if 'd' in Args.a:
			midi['actions'] += mb.DIM_CHORDS
		if 'a' in Args.a:
			midi['actions'] += mb.AUG_CHORDS

	if Args.d:
		print('-----input-----')
		print("Melody: ", midi['states'])
		print("Harmony:", midi['chords'])

	return midi

def setup_agent(Args, Midi):
	if Args.ci or Args.c: # learn with given chords
		rewards = get_rewards(Midi['chords'], Midi['actions'], 1)
		env = Environment(states=Midi['states'], actions=Midi['actions'], rewards=rewards)
		agent = Agent(env)
	else: # learn with TAMER
		env = Environment(states=Midi['states'], actions=Midi['actions'])
		agent = Agent(env)

	return agent

def train(Args, Midi, Agent):
	if Args.ci or Args.c: # learn with given chords
		print('-----training-----')
		for i in range(Args.t):
			if Args.d and Args.t > 9 and (i+1)%int(Args.t/10) == 0:
				print("{0:2.0f}% trained".format(i/Args.t*100))
				print("Weights:", Agent.w)
			Agent.episode(False)

	else: # learn with TAMER
		for i in range(Args.t):
			tamer(Agent, Midi)
			print('-----training-----' + str(i+1) + '/' + str(Args.t))
			for j in range(TAMER_TRIALS):
				Agent.episode(False)

	return Agent

def evaluate(agent, midi, key):
	chords = midi['chords']
	scaleChords = mb.get_scale_chords(key[0], key[1:].lower()=='major')
	scaleNotes = mb.get_scale_notes(key[0], key[1:].lower()=='major')

	agent.explore = False
	agent.evaluate()
	agent.explore = True
	finalActions = [midi['actions'][a] for a in agent.selectedActions]

	correctScore = 0
	for i,action in enumerate(finalActions):
		if mb.chord_eq(action, chords[i]):
			correctScore += 1
	
	functionScore = 0
	npc = np.array(scaleChords)
	tonics = [scaleChords[0], scaleChords[2], scaleChords[5]]
	doms = [scaleChords[4], scaleChords[6]]
	subdoms = [scaleChords[1], scaleChords[3]]
	# print(tonics, doms, subdoms)
	for i,action in enumerate(finalActions):
		if mb.chord_eq(action, chords[i]):
			functionScore += 1
		elif not mb.chord_in(chords[i], scaleChords):
			continue
		else:
			if mb.chord_in(chords[i], tonics) and mb.chord_in(action, tonics):
				functionScore += 0.5
			elif mb.chord_in(chords[i], doms) and mb.chord_in(action, doms):
				functionScore += 0.5
			elif mb.chord_in(chords[i], subdoms) and mb.chord_in(action, tonics):
				functionScore += 0.5

	inkeyScore = 0
	for i,action in enumerate(finalActions):
		if mb.chord_eq(action, chords[i]):
			inkeyScore += 1
		else:
			inkey = 0
			for note in action:
				if note in scaleNotes:
					inkey += 1/len(action)
			inkeyScore += inkey

	correctScore /= len(finalActions)
	functionScore /= len(finalActions)
	inkeyScore /= len(finalActions)

	print("Chord Selection:")
	for i,a in enumerate(finalActions):
		print(chords[i], a)
	print('correct, function, in key')
	print("{0:1.4}\t".format(correctScore) + "{0:1.4}\t".format(functionScore) +
		"{0:1.4}".format(inkeyScore))

	return finalActions, [correctScore, functionScore, inkeyScore]

# play a midi file
def play(file, debug):
	if debug:
		mid = mido.MidiFile(file)
		for msg in mid:
			if msg.type == 'note_on':
				print(mb.ALL_NOTES[(msg.note-21)%12], msg)
			else:
				print(msg)
	pgm.init()
	clock = pgt.Clock()
	pgm.music.load(file)
	pgm.music.play()
	while pgm.music.get_busy():
		clock.tick()

# generates a harmony using the agent's currenty policy and saves it as midi
def generate_output(agent, midi, save, playback, ext, finalActions=[]):
	# get final output from current policy (no exploration)
	if not finalActions:
		agent.explore = False
		agent.evaluate()
		agent.explore = True
		final = agent.selectedActions
		for i,a in enumerate(final):
			finalActions.append(midi['actions'][a])


	print('-----output-----')
	print("Feature Weights:", agent.w)

	print("Chord Selection")
	for i,a in enumerate(finalActions):
		print(a)

	# generate output file & playback
	if 'melodyTrack' in midi:
		midiOut = create_midi(midi, finalActions)
	else:
		midiOut = create_midi(midi, finalActions, midi['states'])

	if save or playback:
		print('saving file as "' + args.m[:-4] + ext + '"')
		midiOut.save(args.m[:-4] + ext)
		if playback:
			play(args.m[:-4] + ext, False)
	return finalActions

# extracts melody and harmony arrays from midi file(s) for training
def process_midi_input(melodyFile, harmonyFile=''):
	mid = mido.MidiFile(melodyFile)
	midi = {'mid': mid, 'melodyFile': melodyFile, 'harmonyFile': harmonyFile}

	# midi meta data
	tpm = 480*4
	for msg in mid:
		if msg.type == 'time_signature':
			# print(msg,mid.ticks_per_beat)
			cpc = msg.clocks_per_click
			tpm = mid.ticks_per_beat*msg.numerator
			midi['tpm'] = tpm
			midi['numerator'] = msg.numerator
			midi['denominator'] = msg.denominator
			midi['cpc'] = cpc
		if msg.type == 'set_tempo':
			midi['tempo'] = msg.tempo

	# melody track selection
	if melodyFile == "Starboy.mid":
		melTrackNum = 2
	elif melodyFile == "Despacito.mid":
		melTrackNum = 3
	elif len(mid.tracks) > 1:
		print("Select Melody Track #")
		for i,track in enumerate(mid.tracks):
			print(str(i) + ":", track.name)
		melTrackNum = int(getch())
	else:
		melTrackNum = 0

	melodyTrack = mid.tracks[melTrackNum]
	
	# harmony track selection
	harmonyTrack = None
	harmTrackNum = melTrackNum
	if melodyFile == "Starboy.mid":
		harmonyTrack = mid.tracks[1]
	elif melodyFile == "Despacito.mid":
		harmonyTrack = mid.tracks[2]
	elif harmonyFile != '':
		midH = mido.MidiFile(harmonyFile)
		if len(midH.tracks) > 1:
			print("Select Harmony Track #")
			for i,track in enumerate(midH.tracks):
				print(str(i) + ":", track.name)
			harmTrackNum = int(getch())
		else:
			harmTrackNum = 0

		harmonyTrack = midH.tracks[harmTrackNum]
	elif args.ci:
		if len(mid.tracks) > 1:
			print("Select Harmony Track #")
			for i,track in enumerate(mid.tracks):
				print(str(i) + ":", track.name)
			harmTrackNum = int(getch())
			
		else:
			harmTrackNum = 0

		harmonyTrack = mid.tracks[harmTrackNum]

	# create array of melody measures
	totalTime = 0
	measure = 0
	states = []
	for i,msg in enumerate(melodyTrack):
		# if args.d:
		# 	print(msg)
		totalTime += msg.time
		measure = int(totalTime/tpm)
		if msg.type == 'note_on':
			while len(states)-1 < measure:
				states.append([])
			states[measure].append(mb.ALL_NOTES[(msg.note-21)%12])

	# create array of harmony measures
	chords = [[] for i in range(len(states))]
	if harmonyTrack:
		totalTime = 0
		measure = 0
		for i,msg in enumerate(harmonyTrack):
			# if args.d:
			# 	print(msg)
			totalTime += msg.time
			measure = int(totalTime/tpm)
			if msg.type == 'note_on':
				# print(len(chords), measure)
				if measure >= len(chords):
					break
				note = mb.ALL_NOTES[(msg.note-21)%12]
				if note not in chords[measure]:
					chords[measure].append(note)
		for i,chord in enumerate(chords):
			if not chord:
				chords[i] = chords[i-1]

	midi['measures'] = len(states)
	midi['chords'] = chords
	midi['melodyTrack'] = melodyTrack
	midi['harmonyTrack'] = harmonyTrack
	midi['states'] = states
	return midi

# extracts melody and harmony arrays from text files for training
def pretrain_midi():
	midi = {}
	midi['melodyFile'] = args.m

	with open('melodies.txt', 'r') as mels:
		lines = mels.read()
		midi['states'] = []
		for row in lines.split('\n'):
			if not row or row[0] == '#':
				continue
			midi['states'].append(row.split(','))
	with open('harmonies.txt', 'r') as harms:
		lines = harms.read()
		midi['chords'] = []
		for row in lines.split('\n'):
			if not row:
				midi['chords'].append([])
			if row[0] != '#':
				midi['chords'].append(row.split(','))

	return midi


# runs one cycle (once through the melody) of tamer training
reward = []
count = 0
keyclock = pgt.Clock()
tamerReward = []
def tamer(agent, midi):
	global reward, count, tamerReward
	# create playback for taming
	agent.evaluate()
	finalActions = []
	for a in agent.selectedActions:
		finalActions.append(agent.env.actions[a])

	finalActions = generate_output(agent,midi,True,False,'-tamer.mid', finalActions)
	tamerFile = midi['melodyFile'][:-4]+'-tamer.mid'

	# setup key press callback
	reward = [0] * midi['measures']
	count = 0
	def keydown(key):
		global reward, count, keyclock
		if 50 > keyclock.tick():
			return
		if key.name == 'delete':
			count -= 1
		elif count >= len(reward):
			return
		elif key.name == 'q':
			reward[count] = -1
			count += 1
		elif key.name == 'w':
			reward[count] = -2
			count += 1
		elif key.name == 'e':
			reward[count] = -3
			count += 1
		elif key.name == '1':
			reward[count] = 1
			count += 1
		elif key.name == '2':
			reward[count] = 2
			count += 1
		elif key.name == '3':
			reward[count] = 3
			count += 1
		elif key.name == 'space':
			reward[count] = 0
			count += 1
		print(reward[0:count], count, '/', len(reward))

	keyboard.on_press(keydown)

	# set up and play tamer midi
	pgm.init()
	clock = pgt.Clock()
	pgm.music.load(tamerFile)
	pgm.music.play()

	while pgm.music.get_busy():
		clock.tick()

	# wait for reward for every measure, add it to environment, run episode
	while(count < len(reward)):
		x = 0
	print('reward given:', reward)
	tamerReward = get_tamer_rewards(finalActions, midi['actions'], reward, tamerReward)
	agent.env.set_rewards(tamerReward)
	agent.episode(True)


# create 2d reward array for given set of chords an actions
def get_rewards(chords, actions, val):
	rewards = [[0]*len(actions) for i in range(len(chords))]
	for i in range(len(chords)):
		for j in range(len(actions)):
			allThere = True
			for note in chords[i]:
				if note not in actions[j]:
					allThere = False
			if allThere:
				rewards[i][j] = val
	return rewards

# modify 2d reward array with new tamer reinforcement
def get_tamer_rewards(chords, actions, reinforcement, rewards):
	assert(len(chords) == len(reinforcement))
	if not rewards:
		rewards = [[0]*len(actions) for i in range(len(chords))]
	for i in range(len(chords)):
		for j in range(len(actions)):
			allThere = True
			for note in chords[i]:
				if note not in actions[j]:
					allThere = False
			if allThere:
				rewards[i][j] = reinforcement[i]
	return rewards

# create midi file combining melody and chords
def create_midi(midi, chords, melody=None):
	mid = mido.MidiFile()
	mid.ticks_per_beat = int(midi['tpm']/4)
	if 'melodyTrack' in midi:
		mid.tracks.append(midi['melodyTrack'])
	else:
		mid.tracks.append(create_midi_track(melody, True, midi['tpm'], midi['tempo']))

	mid.tracks.append(create_midi_track(chords, False, midi['tpm'], midi['tempo']))
	return mid

# create_midi helper, creates a single track.
# TODO: fix melody timing so notes aren't evenly dispersed
def create_midi_track(notes, isMelody, tpm, tempo=1):
	track = mido.MidiTrack()

	name = 'RL Harmony'
	if isMelody:
		name = 'RL Melody'
	track.append(mido.MetaMessage('track_name', name=name, time=0))

	if tempo != 1:
		track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

	if isMelody:
		for i,run in enumerate(notes):
			for note in run:
				noteNum = (mb.ALL_NOTES.index(note)+21)+36
				track.append(mido.Message('note_on', note=noteNum, velocity=64, time=0))
				track.append(mido.Message('note_off', note=noteNum, velocity=64, time=int(tpm/len(run))))
	else:
		for i,run in enumerate(notes):
			for note in run:
				noteNum = (mb.ALL_NOTES.index(note)+21)+36
				track.append(mido.Message('note_on', note=noteNum, velocity=64, time=0))
			for j,note in enumerate(run):
				noteNum = (mb.ALL_NOTES.index(note)+21)+36
				if j == 0:
					track.append(mido.Message('note_off', note=noteNum, velocity=64, time=tpm))
				else:
					track.append(mido.Message('note_off', note=noteNum, velocity=64, time=0))
	return track


# get a character from command line
# source: https://code.activestate.com/recipes/577977-get-single-keypress/
try:
	import tty, termios
except ImportError:
	# Probably Windows.
	try:
		import msvcrt
	except ImportError:
		# FIXME what to do on other platforms?
		# Just give up here.
		raise ImportError('getch not available')
	else:
		getch = msvcrt.getch
else:
	def getch():
		"""getch() -> key character

		Read a single keypress from stdin and return the resulting character. 
		Nothing is echoed to the console. This call will block if a keypress 
		is not already available, but will not wait for Enter to be pressed. 

		If the pressed key was a modifier key, nothing will be detected; if
		it were a special function key, it may return the first character of
		of an escape sequence, leaving additional characters in the buffer.
		"""
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(fd)
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch


if __name__ == '__main__':
	main()





