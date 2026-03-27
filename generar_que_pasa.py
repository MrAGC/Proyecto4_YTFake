import pandas as pd
import random

# Cargar dataset
df = pd.read_csv("videos.csv")

# 100 palabras reales por categoría
category_words = {
    "Gaming": [
        "fortnite","minecraft","callofduty","valorant","csgo","apex","leagueoflegends","roblox","eldenring","fifa",
        "gameplay","stream","multiplayer","online","ranking","fps","shooter","battle","royale","loot",
        "skin","weapon","map","mission","quest","level","xp","boss","enemy","strategy",
        "teclado","raton","controller","console","pc","setup","esports","tournament","team","clan",
        "sniper","headshot","kills","deathmatch","arena","match","victory","defeat","stats","score",
        "update","patch","meta","build","inventory","craft","survival","sandbox","openworld","rpg",
        "speedrun","glitch","mods","dlc","campaign","story","cutscene","graphics","fpsboost","lag",
        "ping","server","crossplay","coop","solo","ranked","casual","achievement","trophy","unlock",
        "aim","reaction","skill","practice","training","guide","tutorial","tips","tricks","challenge",
        "event","season","battlepass","cosmetics","avatar","character","loadout","upgrade","perk","ability"
    ],

    "Music": [
        "song","track","album","artist","band","concert","live","tour","festival","stage",
        "melody","rhythm","beat","lyrics","voice","vocals","instrumental","guitar","piano","drums",
        "bass","singer","rapper","dj","producer","mix","remix","studio","recording","sound",
        "genre","pop","rock","hiphop","rap","jazz","classical","electronic","house","techno",
        "spotify","playlist","stream","chart","hit","single","release","cover","version","acoustic",
        "performance","microphone","speaker","audio","headphones","volume","tempo","harmony","tune","note",
        "chorus","verse","hook","drop","bridge","sample","loop","synth","autotune","effect",
        "fan","crowd","ticket","show","backstage","gig","label","contract","industry","fame",
        "viral","trend","youtube","video","clip","mv","dance","vibe","mood","energy",
        "emotion","feeling","expression","art","creative","style","culture","inspiration","collab","feature"
    ],

    "Sports": [
        "football","basketball","tennis","golf","baseball","soccer","rugby","boxing","mma","cycling",
        "race","marathon","swimming","olympics","league","tournament","cup","final","match","game",
        "team","player","coach","training","practice","fitness","strength","speed","agility","endurance",
        "goal","score","point","win","lose","draw","victory","defeat","champion","medal",
        "stadium","field","court","arena","track","fans","crowd","support","cheer","club",
        "transfer","contract","salary","season","schedule","ranking","table","stats","performance","highlight",
        "injury","recovery","physio","warmup","cooldown","diet","nutrition","hydration","rest","sleep",
        "referee","foul","penalty","card","rule","fairplay","offside","kick","pass","dribble",
        "shoot","defense","attack","strategy","formation","tactics","analysis","review","replay","broadcast",
        "coach","captain","substitute","bench","lineup","selection","competition","qualifier","finals","trophy"
    ],

    "Tech": [
        "technology","software","hardware","computer","laptop","mobile","smartphone","tablet","device","gadget",
        "app","application","program","code","coding","programming","developer","engineer","system","platform",
        "ai","machinelearning","deeplearning","neuralnetwork","data","bigdata","analytics","cloud","server","database",
        "internet","web","website","browser","network","wifi","bluetooth","5g","iot","security",
        "cybersecurity","encryption","privacy","password","login","authentication","firewall","virus","malware","antivirus",
        "update","upgrade","version","release","beta","testing","debug","bug","fix","patch",
        "interface","ui","ux","design","frontend","backend","fullstack","api","integration","automation",
        "robot","drone","vr","ar","metaverse","blockchain","crypto","bitcoin","nft","fintech",
        "startup","innovation","future","digital","transformation","scalability","performance","optimization","efficiency","tool",
        "framework","library","open source","github","repository","deployment","hosting","docker","kubernetes","pipeline"
    ],

    "Education": [
        "education","school","university","college","student","teacher","professor","class","lesson","lecture",
        "study","learning","knowledge","subject","topic","course","degree","exam","test","quiz",
        "homework","assignment","project","research","thesis","paper","presentation","notes","reading","writing",
        "math","science","history","geography","biology","chemistry","physics","language","literature","art",
        "online","elearning","platform","video","tutorial","guide","explained","example","exercise","practice",
        "skills","criticalthinking","problem","solution","analysis","logic","memory","focus","attention","discipline",
        "schedule","routine","planning","organization","productivity","goal","achievement","progress","improvement","feedback",
        "discussion","debate","question","answer","explanation","clarity","understanding","concept","theory","model",
        "experiment","lab","data","result","observation","conclusion","method","approach","strategy","technique",
        "career","future","job","training","internship","experience","development","growth","success","opportunity"
    ],

    "Comedy": [
        "comedy","humor","funny","joke","laugh","laughter","meme","sketch","parody","satire",
        "sarcasm","irony","giggle","smile","hilarious","ridiculous","absurd","weird","random","silly",
        "prank","reaction","fails","epicfail","awkward","moment","clip","viral","trend","internet",
        "youtube","tiktok","short","video","content","creator","influencer","audience","view","like",
        "share","comment","subscribe","channel","episode","series","show","character","story","scene",
        "dialogue","punchline","timing","delivery","expression","face","gesture","voice","tone","style",
        "improv","standup","performance","stage","crowd","energy","act","routine","material","script",
        "spoof","mock","exaggeration","twist","surprise","unexpected","crazy","wild","fun","entertainment",
        "laughing","crying","rolling","floor","reaction","clip","moment","highlight","best","top",
        "compilation","collection","edit","montage","remix","edit","sound","effect","music","background"
    ],

    "Lifestyle": [
        "lifestyle","life","routine","daily","morning","night","habits","health","wellness","fitness",
        "exercise","gym","workout","diet","nutrition","food","meal","recipe","cooking","kitchen",
        "home","house","apartment","decor","design","style","fashion","outfit","clothes","shopping",
        "beauty","skincare","makeup","hair","selfcare","relax","mindfulness","meditation","yoga","balance",
        "travel","trip","vacation","holiday","destination","adventure","explore","nature","city","culture",
        "friends","family","relationship","social","community","events","party","celebration","weekend","fun",
        "budget","money","saving","spending","minimalism","organization","cleaning","productivity","planning","goal",
        "motivation","inspiration","growth","development","mindset","success","happiness","joy","gratitude","positivity",
        "work","career","office","remote","freelance","business","entrepreneur","sidehustle","project","focus",
        "sleep","rest","energy","routine","schedule","time","management","efficiency","balance","lifestyle"
    ],

    "News": [
        "news","breaking","update","report","journalism","media","press","headline","story","coverage",
        "world","global","international","local","national","politics","government","election","policy","law",
        "economy","finance","market","stock","inflation","crisis","growth","trade","business","industry",
        "technology","science","health","environment","climate","weather","disaster","emergency","alert","warning",
        "event","incident","accident","conflict","war","protest","movement","society","culture","community",
        "interview","statement","analysis","opinion","editorial","commentary","insight","perspective","fact","data",
        "investigation","evidence","reporting","source","information","truth","fake","misinformation","debate","discussion",
        "live","stream","broadcast","channel","tv","radio","online","platform","social","network",
        "viral","trend","reaction","public","response","impact","effect","change","development","timeline",
        "history","context","background","detail","summary","highlight","recap","latest","today","now"
    ]
}

# Función: elegir 5 palabras aleatorias
def generate_words(category):
    words = category_words.get(category, ["general", "contenido", "video", "tema", "random"])
    return ", ".join(random.sample(words, 5))

# Crear columna
df["que_pasa"] = df["category"].apply(generate_words)

# Guardar
df.to_csv("videos_actualizado.csv", index=False)

print("✅ Listo: 100 palabras por categoría y 5 por fila")