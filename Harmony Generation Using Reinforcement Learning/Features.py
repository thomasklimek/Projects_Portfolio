import MusicBank as mb


def all_features():
	feats = []
	feats.append(count_feature)
	feats.append(shared_feature)
	feats.append(tritone_feature)
	feats.append(last_bar_feature)

	# no good
	# feats.append(root_feature)
	# feats.append(third_feature)
	# feats.append(mode_feature)
	# feats.append(previous_same_feature)
	# feats.append(previous_scale_feature)
	# feats.append(previous_2_scale_feature)
	return feats


def calc_features(env, time, action, recurse=False):
	featureVals = [0] * len(env.features)
	for i,f in enumerate(env.features):
		if not recurse or f != last_bar_feature:
			featureVals[i] = f(env, time, action)
	# print('Features', str(env.actions[action]), env.states[time], str(features))
	return featureVals

def last_bar_feature(env, time, action):
	if time < 1:
		return 0
	lastVals = calc_features(env, time-1, action, True)
	return sum(lastVals)/len(lastVals)

# pct of state notes in the action
def count_feature(env, time, action):
	aNotes = env.actions[action]
	sNotes = env.states[time]
	if not sNotes:
		return 0
	count = 0
	for note in sNotes:
		if note in aNotes:
			count += 1
	return count / len(sNotes)

# pct of action notes in the state
def shared_feature(env, time, action):
	aNotes = env.actions[action]
	sNotes = env.states[time]
	if not sNotes:
		return 0
	count = 0
	for note in aNotes:
		if note in sNotes:
			count += 1
	return count / len(aNotes)

# 1 if the root of the action is in the state
def root_feature(env, time, action):
	aNotes = env.actions[action]
	sNotes = env.states[time]
	if not sNotes:
		return 0
	if aNotes[0] in sNotes:
		return 1
	else:
		return 0

# 1 if third of the action is in the state
def third_feature(env, time, action):
	aNotes = env.actions[action]
	sNotes = env.states[time]
	if not sNotes:
		return 0
	if aNotes[1] in sNotes:
		return 1
	else:
		return 0

# 1 if action doesn't have a tritone with any state note
def tritone_feature(env, time, action):
	aNotes = env.actions[action]
	sNotes = env.states[time]
	if not sNotes:
		return 0
	for aNote in aNotes:
		tritone = mb.get_interval(aNote, 6)
		if tritone in sNotes:
			return 0
	return 1

def mode_feature(env, time, action):
	aNotes = env.actions[action]
	sNotes = env.states[time]
	if not sNotes:
		return 0

	mode = max(sNotes, key=aNotes.count)
	if mode in sNotes:
		return 1
	return 0

# are action and last action in each other's scale
def previous_scale_feature(env, time, action):
	if env.time == 0:
		return 0

	aNotes = env.actions[action]
	pNotes = env.actions[env.actionHistory[time-1]]

	aName = mb.get_chord_name(aNotes)
	pName = mb.get_chord_name(pNotes)

	aScale = mb.get_scale_chords(aName[0], aName[1:]=='major')
	pScale = mb.get_scale_chords(pName[0], pName[1:]=='major')

	score = 0
	if mb.chord_in(aNotes, pScale):
		score += 0.5
	if mb.chord_in(pNotes, aScale):
		score += 0.5

	return score

# are action and last 2 actions in a scale
def previous_2_scale_feature(env, time, action):
	if env.time < 2:
		return 0

	aNotes = env.actions[action]
	pNotes = env.actions[env.actionHistory[time-1]]
	ppNotes = env.actions[env.actionHistory[time-2]]

	
	for note in mb.ALL_NOTES:
		major = mb.get_scale_chords(note, True)
		if (mb.chord_in(aNotes, major) and 
		mb.chord_in(pNotes, major) and
		mb.chord_in(ppNotes, major)):
			return 1

		minor = mb.get_scale_chords(note, False)
		if (mb.chord_in(aNotes, minor) and 
		mb.chord_in(pNotes, minor) and
		mb.chord_in(ppNotes, minor)):
			return 1

	return 0

def previous_same_feature(env, time, action):
	if env.time < 2:
		return 0

	aNotes = env.actions[action]
	pNotes = env.actions[env.actionHistory[time-1]]

	same = 1
	if mb.chord_eq(aNotes, pNotes):
		same = 0
	return same


# def interval_feature(env, time, action):
# 	lastAction = env.actionHistory[time-1]
# 	lastNotes = env.actions[lastAction]
# 	aNotes = env.action_notes(action)

# 	isMajor = aNotes in mb.MAJOR_CHORDS
# 	aRoot = aNotes[0]

# 	possChords = []
# 	possChords += mb.scale_chords(aRoot, isMajor)