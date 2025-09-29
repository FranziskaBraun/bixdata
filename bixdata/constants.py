SEED = 42
# , '[.....]', '[......]', '[.......]'
SPECIAL_TOKENS = ['[..]', '[...]', '[....]', '[*]']
TASK_MAPPING = {'h6uxbwun': 'Johanna_Subway',
                'ohkj1l71': 'table',
                '7v6k33ji': 'window',
                'b4pglcxp': 'bowl',
                'if1c1rts': 'hanger',
                'w7bvn6os': 'fir_tree',
                'w3hyvkzn': 'mountains',
                '5e0rjyrv': 'comb',
                'vvbccyv8': 'flute',
                'fnn9b45p': 'golf_club',
                'q4kgsuym': 'boat',
                '16e75yco': 'turtle',
                'nxw2ivk4': 'violin',
                'zpfcg8d0': 'camper_van',
                'pbw7wlzs': 'wisk',
                'u0daanxn': 'key',
                'qmszig1h': 'leaf',
                'zbuvcfzi': 'dice',
                '04lxopuv': 'category_animal',
                '80ui0t50': 'picture_description_mountain_scene',
                'n91euyp5': 'pataka_repeat', 'gjin33l3': 'sischafu_repeat',
                'yt1y8hou': 'recall_story',
                'u7kztp6d': 'recall_picture_description',
                'bnt': 'boston_naming_test'}

BNT = (
'w7bvn6os', 'ohkj1l71', 'vvbccyv8', 'qmszig1h', '7v6k33ji', 'q4kgsuym', '5e0rjyrv', 'w3hyvkzn', 'u0daanxn', '16e75yco',
'nxw2ivk4', 'pbw7wlzs', 'zpfcg8d0', 'if1c1rts', 'zbuvcfzi')

BIX_BNT_ANSWER = 'Tanne Tisch Flöte Blatt Fenster Boot Kamm Berge Schlüssel Schildkröte Geige Schneebesen Wohnmobil Kleiderbügel Würfel'
MC_BNT_ANSWER = 'Baum Bett Pfeife Blume Haus Kanu Zahnbürste Vulkan Maske Kamel Mundharmonika Zange Hängematte Trichter Dominosteine'

SAMPLING_RATE = 16000
MAX_AUDIO_DURATION_S = 150
FRAME_RATE_MS = 10
BEAM_SIZE = 5