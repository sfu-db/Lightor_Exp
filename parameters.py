"""
Hyperprarameters for Highlight Initializer
"""
INITIAL_PARA = {'interval_time': 5, 'latency': 10, 'num': 200}
EXPAND = {'bin_size': 3, 'win_size': 7, 'polyorder': 4}

"""
Hyperprarameters for Highlight Extractor
"""

FILTER_INTERVAL = 60
SELECTION_THRESHOLD = 20
ERROR_THRESHOLD = 40
GRAPH_LOWER_BOUND_LENGTH = 4
FEATURE_VECTOR = [[4, 33, 252], [25, 12, 56]]


"""
Constant for crowdsourcing eXP
"""

"""
This is the initial red dots positions set in the first round of crowdsourcing exp,
which are generated from Highlight Initializer. The key numbers are the ID of video clips
we cut from test videos.
"""
ORIGINAL_DATA = {'filtered': {},
 'remaining': {'714': [[2103, -1]],
  '715': [[9, -1]],
  '716': [[3266, -1]],
  '717': [[1389, -1]],
  '718': [[1816, -1]],
  '719': [[2607, -1]],
  '720': [[249, -1]],
  '721': [[1395, -1]],
  '722': [[2914, -1]],
  '723': [[3109, -1]],
  '724': [[5707, -1]],
  '725': [[5480, -1]],
  '726': [[6593, -1]],
  '727': [[1741, -1]],
  '728': [[7001, -1]],
  '734': [[3408, -1]],
  '735': [[2364, -1]],
  '736': [[250, -1]],
  '737': [[1807, -1]],
  '738': [[1169, -1]],
  '739': [[1667, -1]],
  '740': [[1375, -1]],
  '741': [[4133, -1]],
  '742': [[4601, -1]],
  '743': [[696, -1]],
  '749': [[1687, -1]],
  '750': [[710, -1]],
  '751': [[234, -1]],
  '752': [[1973, -1]],
  '753': [[1513, -1]],
  '754': [[2162, -1]],
  '755': [[1315, -1]],
  '756': [[1973, -1]],
  '757': [[808, -1]],
  '758': [[1828, -1]]},
 'selected': {}}


"""
This is used to map video clips to original test videos, which are used in evaluation
"""

DOTA2_MOVIE_NAME_BY_ID = {'714': 'attackerdota-2017-07-31-02h34m10s',
 '715': 'attackerdota-2017-07-31-02h34m10s',
 '716': 'attackerdota-2017-07-31-02h34m10s',
 '717': 'attackerdota-2017-07-31-02h34m10s',
 '718': 'attackerdota-2017-07-31-02h34m10s',
 '719': 'attackerdota-2017-07-31-01h34m09s',
 '720': 'attackerdota-2017-07-31-01h34m09s',
 '721': 'attackerdota-2017-07-31-01h34m09s',
 '722': 'attackerdota-2017-07-31-01h34m09s',
 '723': 'attackerdota-2017-07-31-01h34m09s',
 '724': 'moonmeander-2017-07-03-19h17m57s',
 '725': 'moonmeander-2017-07-03-19h17m57s',
 '726': 'moonmeander-2017-07-03-19h17m57s',
 '727': 'moonmeander-2017-07-03-19h17m57s',
 '728': 'moonmeander-2017-07-03-19h17m57s',
 '729': 'moonmeander-2017-07-03-19h17m57s',
 '730': 'moonmeander-2017-07-03-19h17m57s',
 '731': 'moonmeander-2017-07-03-19h17m57s',
 '732': 'moonmeander-2017-07-03-19h17m57s',
 '733': 'moonmeander-2017-07-03-19h17m57s',
 '734': 'sing_sing-2017-08-01-10h09m35s',
 '735': 'sing_sing-2017-08-01-10h09m35s',
 '736': 'sing_sing-2017-08-01-10h09m35s',
 '737': 'sing_sing-2017-08-01-10h09m35s',
 '738': 'sing_sing-2017-08-01-10h09m35s',
 '739': 'moonmeander-2017-07-03-17h17m56s',
 '740': 'moonmeander-2017-07-03-17h17m56s',
 '741': 'moonmeander-2017-07-03-17h17m56s',
 '742': 'moonmeander-2017-07-03-17h17m56s',
 '743': 'moonmeander-2017-07-03-17h17m56s',
 '744': 'moonmeander-2017-07-03-17h17m56s',
 '745': 'moonmeander-2017-07-03-17h17m56s',
 '746': 'moonmeander-2017-07-03-17h17m56s',
 '747': 'moonmeander-2017-07-03-17h17m56s',
 '748': 'moonmeander-2017-07-03-17h17m56s',
 '749': 'attackerdota-2017-07-30-23h34m09s',
 '750': 'attackerdota-2017-07-30-23h34m09s',
 '751': 'attackerdota-2017-07-30-23h34m09s',
 '752': 'attackerdota-2017-07-30-23h34m09s',
 '753': 'attackerdota-2017-07-30-23h34m09s',
 '754': 'sing_sing-2017-08-01-09h09m34s',
 '755': 'sing_sing-2017-08-01-09h09m34s',
 '756': 'sing_sing-2017-08-01-09h09m34s',
 '757': 'sing_sing-2017-08-01-09h09m34s',
 '758': 'sing_sing-2017-08-01-09h09m34s'}
