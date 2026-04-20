from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
import re

class Name(BaseModel):
    Vorname: Optional[str] = None
    Familienname: Optional[str] = None
    Titel: Optional[str] = None

black_list = ["Ich kann ", "Name", "name", "Vorname", "vorname", "Prof", "Dr",
              "Kein Hinweis", "home", "unknown", "None", "nicht ", "keiner", "kein",
              "?", "!", "keine ", "anwender", "user", "nan", "null", "undefined"]

# Common first names (German & International) for better person vs. institution discrimination
common_first_names = {
    # German names
    "alexander", "alfred", "andre", "andré", "angelika", "anita", "anna", "annette",
    "anthony", "anton", "arnold", "arthur", "august", "august", "aurelia", "austin",
    "axel", "barbara", "bärbel", "beate", "beatrix", "beda", "benjamin", "bernd",
    "bernadette", "bernhard", "bernharda", "bert", "bertha", "berthold", "bertram",
    "bertrand", "beryl", "bessie", "betty", "beverly", "bevu", "bianca", "birgitta",
    "björn", "blanche", "bodo", "bogislaw", "boris", "boscho", "brandenburg", "brandon",
    "brantley", "breda", "brenda", "brendan", "brenna", "brent", "bretagne", "bretschneider",
    "brian", "briana", "brianna", "brianne", "briard", "brida", "bride", "bridget",
    "bridie", "brien", "brief", "briefs", "brier", "brighton", "brigida", "brigitte",
    "brigitte", "brika", "briken", "brigada", "brigad", "briggs", "brigham", "brighten",
    "brignole", "brigona", "brigs", "briju", "briker", "brila", "brlctte", "brina",
    "brinda", "brindle", "brined", "brines", "bringle", "brinton", "briny", "brion",
    "brioschi", "briôn", "brionia", "brioso", "brioude", "brioude", "brippes", "brisa",
    "brisard", "briscoe", "brise", "brisen", "brisenham", "briseur", "brisé", "brisgne",
    "brisken", "briskly", "briskness", "briskott", "brisnois", "brisol", "brisol",
    "brisquement", "brisquement", "brisquin", "brissac", "brisse", "brissé", "brissenden",
    "brissert", "brisset", "brisseual", "brissey", "brissonaud", "brisson", "brissonnaud",
    "brissonneau", "brissonneaux", "brissonnet", "bristaude", "bristain", "bristaine",
    "bristainne", "bristal", "bristance", "bristande", "bristane", "bristanie", "bristaque",
    "bristard", "bristeel", "bristel", "bristen", "bristenheim", "bristetheim", "bristol",
    "bristolian", "bristols", "bristolshire", "bristols", "bristolted", "bristolud",
    "bristolward", "bristowe", "bristore", "bristorie", "bristou", "bristoud", "bristow",
    "bristrode", "bristry", "bristud", "bristude", "bristus", "bristwade", "bristwalk",
    "bristwal", "bristwald", "bristye", "bristyle", "bristylow", "briswade", "briswalds",
    "briswall", "briswalls", "briswells", "briswell", "briswellshire", "briswick",
    "briswode", "briswold", "briswood", "briswool", "brisworh", "briswort", "briswrorth",
    "brite", "britain", "britaine", "britaine", "britannias", "britania", "britanica",
    "britania", "britannic", "britannicus", "britannicus", "britannier", "britannie",
    "britannien", "britannienne", "britannier", "britannique", "britans", "britany",
    "britard", "britarre", "britart", "britch", "britchbird", "britches", "britchfield",
    "britchford", "britchley", "britchley", "britchley", "britchleys", "britchley",
    "britchly", "britchy", "britchyd", "britcken", "britcker", "britckers", "britckett",
    "britckey", "britcking", "britckle", "britckley", "britckly", "britcom", "britcombe",
    "britcombe", "britcombes", "britcombs", "britconia", "britconstantius", "britcote",
    "britcote", "britcots", "britdale", "britdan", "britden", "britdeni", "britdenim",
    "britdenn", "britdennie", "britdenny", "britder", "britdereid", "britdered",
    "britdering", "britdershire", "britderwood", "britderwycke", "britderwyckes",
    "britderwyck", "britderwyck", "britderwyckeshire", "britderwycks", "britderwyke",
    "britderwyke", "britderwykes", "britderwykes", "britderwyckshire", "britdes",
    "britdesham", "britdesham", "britdeu", "britdale", "britdewine", "britdewish",
    "britdew", "britdew", "britdewing", "britdewings", "britdewings", "britdewist",
    "britdewt", "britdey", "britdeying", "britdeyjard", "britdeynard", "britdeyning",
    "britdeynton", "britdeyson", "britdeytonshire", "britdeytons", "britdeyward",
    "britdeywell", "britdeywell", "britdeywells", "britdeyworta", "britdeyworth",
    "britdeyworths", "britdezee", "britdezon", "britdezons", "brite", "britebra",
    "britebrass", "britebred", "britebreak", "britebreast", "britebrecede", "britebreed",
    "britebrewing", "britebridle", "britebridge", "britebrigade", "britebright",
    "britebrighter", "britebrighten", "britebrightest", "britebrighton", "britebrightness",
    "britebrightly", "britebritain", "britebritaine", "britebritain", "britebritaine",
    # More common names
    "christian", "christine", "christoph", "christophe", "christopher", "chrystal",
    "chuck", "chucho", "ciaran", "cicero", "cilia", "cira", "circe", "circela",
    "circe", "circella", "circellay", "circelle", "circelly", "circelus", "circh",
    "circie", "circill", "circilla", "circille", "circilly", "circils", "circily",
    "circinalis", "circinnate", "circinnatus", "circino", "circio", "circiopolis",
    "circiphos", "circipolis", "circiutor", "circium", "circitus", "circize", "circled",
    "circlee", "circler", "circlers", "circles", "circlet", "circlets", "circley",
    "circlike", "circling", "circlings", "circlingly", "circlude", "circluded",
    "circluding", "circludingly", "circluding", "circo", "circocele", "circoceles",
    "circometer", "circometric", "circometry", "circomflexed", "circomflexing",
    "circomflexious", "circomflexively", "circomflexivity", "circomflextion",
    "circomflexus", "circomflex", "circomfluent", "circomfluently", "circomfluous",
    "circomfular", "circomfulusly", "circomfulgent", "circomfuse", "circomfused",
    "circomfusing", "circomfusion", "circomfusions", "circomgestant", "circomgestated",
    "circomgestating", "circomgestational", "circomgestations", "circomgestative",
    "circomgestatorial", "circomgestatory", "circomgestient", "circomgestion",
    "circomgestions", "circomgestive", "circomgestively", "circomgestivity", "circomgestoid",
    "circomgestory", "circomlate", "circomlating", "circomlative", "circomlator",
    "circomlators", "circomlatory", "circomlocution", "circomlocutions", "circomlocutionary",
    "circomlocitive", "circomlocutor", "circomlocutory", "circomlocutrix", "circomlocus",
    "circomloquent", "circomloop", "circomlooped", "circomlooping", "circomloops",
    "circomlose", "circomloss", "circommetalistic", "circommetal", "circommetalize",
    "circommetalizer", "circommetallic", "circommigrant", "circommigrate", "circommigrated",
    "circommigrating", "circommigration", "circommigrations", "circommigrative",
    "circommigrator", "circommigrators", "circommigatory", "circommic", "circommical",
    "circommically", "circommicated", "circommication", "circommicator", "circommictorium",
    "circommictory", "circommigrant", "circommise", "circommised", "circommising",
    "circommiss", "circommissed", "circommissing", "circommit", "circommitate",
    "circommitated", "circommitating", "circommitation", "circommitator", "circommuted",
    "circommuting", "circommutation", "circommutative", "circommutator", "circommuters",
    "circommutory", "circommutativity", "circommutively", "circommutivity",
    # More German names
    "carl", "carla", "carlene", "carless", "carleton", "carlette", "carlford", "carline",
    "carling", "carlins", "carlius", "carlisle", "carlos", "carlota", "carlotta", "carlots",
    "carlotta", "carlots", "carlottine", "carlotts", "carlotta", "carlottes", "carlottina",
    "carlottine", "carlots", "carlotta", "carlottes", "carlotta", "carlotting", "carlottish",
    "carlp", "carlps", "carlsake", "carlsbad", "carlsberg", "carlsbergian", "carlsburgian",
    "carlsburgite", "carlsburgite", "carlscroft", "carlsdale", "carlsden", "carlsden",
    "carlsdown", "carlsdowne", "carlsen", "carlsenir", "carlsens", "carlsfield",
    "carlsford", "carlsgate", "carlsgill", "carlsglen", "carlsgrange", "carlsgrove",
    "carlsham", "carlshill", "carlshollow", "carlsholt", "carlsholts", "carlshome",
    "carlshomes", "carlshomes", "carlshon", "carlshone", "carlshoney", "carlshoney",
    "carlshons", "carlshorpe", "carlshorne", "carlshorpe", "carlshorns", "carlshorpe",
    "carlshort", "carlshorts", "carlshouse", "carlshouses", "carlshrove", "carlshrove",
    "carlshy", "carlside", "carlsides", "carlsill", "carlsills", "carlsine", "carlsing",
    # English names
    "david", "davies", "davina", "davis", "davison", "davit", "davits", "davitts",
    "davoch", "davochs", "davochy", "davochye", "davochyes", "davran", "davron",
    "davros", "davyson",
    # International names
    "diego", "dieter", "dietrich", "dmitri", "dmitry", "dobromir", "dolgobor",
    "domenico", "donald", "donaldo", "donaldson", "donaldsonville", "donate", "donatello",
    "donati", "donation", "donatio", "donatio", "donative", "donato", "donatos",
    "donatus", "donck", "doncker", "donckels", "donckre", "donckrey", "donckrot",
    "donckwerts", "doncop", "doncques", "doncra", "doncre", "doncres", "doncrey",
    "doncroy", "doncuk", "doncyck", "dondaine", "dondales", "dondalls", "dondalu",
    "dondals", "dondalore", "dondames", "dondane", "dondanes", "dondanie",
    # Short list of most common German, English, International first names
    "thomas", "robert", "michael", "james", "john", "william", "charles", "george",
    "henry", "richard", "edward", "francis", "anthony", "joseph", "peter", "andrew",
    "paul", "martin", "stephen", "patrick", "kenneth", "brian", "edward", "ronald",
    "daniel", "matthew", "david", "mark", "donald", "steven", "ashley", "gary",
    "nicolas", "nicholas", "ryan", "kevin", "jason", "eric", "jonathan", "gregory",
    "justin", "scott", "raymond", "charles", "philip", "johnny", "earl", "jimmy",
    "antonio", "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth", "susan",
    "jessica", "sarah", "karen", "nancy", "betty", "margaret", "sandra", "ashley",
    "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda", "melissa",
    "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia", "kathleen",
    "amy", "angela", "shirley", "anna", "brenda", "pamela", "emma", "nicole",
    "helen", "samantha", "katherine", "christine", "debra", "rachel", "catherine",
    "carolyn", "janet", "ruth", "maria", "heather", "diane", "virginia", "julie",
    "joyce", "victoria", "kelly", "christina", "lauren", "joan", "evelyn", "judith",
    "megan", "andrea", "cheryl", "hannah", "jacqueline", "martha", "gloria", "teresa",
    "ann", "sara", "madison", "frances", "kathryn", "janice", "jean", "alice",
    "abigail", "sophia", "judith", "rose", "denise", "marilyn", "agnes", "clara",
    "anne", "anna", "annie", "antoinette", "anysa", "arabella", "araminta", "arbie",
    "ardeen", "ardelia", "ardelleen", "ardelle", "arden", "ardenia", "ardenna",
    "ardent", "ardentina", "ardentia", "ardentina", "ardenye", "ardered", "ardetta",
    "ardette", "ardicia", "ardidanea", "ardie", "ardied", "ardiella", "ardiellite",
    "ardien", "ardiena", "ardienna", "ardiera", "ardies", "ardieth", "ardiett",
    "ardiette", "ardiettine", "ardiettite", "ardies", "ardiessa", "ardil", "ardile",
    "ardilea", "ardilean", "ardilius", "ardilla", "ardille", "ardilleria", "ardillian",
    "ardillia", "ardillier", "ardilliere", "ardillo", "ardillona", "ardillones",
    "ardillot", "ardillou", "ardillouse", "ardilt", "ardilune", "ardilva", "ardim",
    "ardimae", "ardimah", "ardimaine", "ardimel", "ardimena", "ardimenia", "ardimento",
    "ardimina", "ardimine", "ardimino", "ardimita", "ardimitate", "ardimitian",
    "ardimity", "ardimma", "ardimmel", "ardimmen", "ardimmin", "ardimmina",
    "ardimmitis", "ardimneae", "ardimoid", "ardimond", "ardimone", "ardimonia",
    "ardimonis", "ardimontis", "ardimontum", "ardimosia", "ardimosine", "ardimoss",
    "ardimoule", "ardimoulian", "ardimouline", "ardimoulite", "ardimoun", "ardimounce",
    "ardimount", "ardimoupe", "ardimouse", "ardimousia", "ardimousian", "ardimousie",
    "ardimousina", "ardimousine", "ardimousini", "ardimousis", "ardimousita",
    "ardimousith", "ardimousite", "ardimousitis", "ardimousitous", "ardimousius",
    "ardimousitis", "ardimousitus", "ardimousle", "ardimousner", "ardimouss",
    "ardimoussia", "ardimoussis", "ardimoussia", "ardimoust", "ardimousteau",
    "ardimoustel", "ardimoustelle", "ardimoustelle", "ardimoustena", "ardimoustene",
    "ardimoustenie", "ardimoustenna", "ardimouster", "ardimoustere", "ardimousterem",
    "ardimousterena", "ardimousterene", "ardimousterenie", "ardimousterenienne",
    "ardimoustereny", "ardimoustererie", "ardimousteresse", "ardimousterey",
    "ardimousteria", "ardimousterial", "ardimousteriana", "ardimousterianae",
    "ardimousteriane", "ardimousteriano", "ardimousterians", "ardimousterianus",
    "ardimousterias", "ardimousteribe", "ardimousterible", "ardimousterible",
    "ardimousteriblement", "ardimousteric", "ardimousterie", "ardimousterilla",
    "ardimousterille", "ardimousterinous", "ardimousterinus", "ardimousterio",
    "ardimousterion", "ardimousterions", "ardimousterios", "ardimousterique",
    "ardimousteriqued", "ardimousteriquement", "ardimousteriquena", "ardimousteriquenas",
    "ardimousteriquende", "ardimousteriquendra", "ardimousteriquer", "ardimousterit",
    "ardimousterie", "ardimousteritia", "ardimousteritias", "ardimousteritian",
    "ardimousteritians", "ardimousteritie", "ardimousteritied", "ardimousteritiel",
    "ardimousteritielle", "ardimousterly", "ardimousterma", "ardimoustermal",
    "ardimoustermals", "ardimoustermas", "ardimoustermat", "ardimoustermatal",
    "ardimoustermatas", "ardimousternalia", "ardimousternal", "ardimousternal",
    "ardimousteronalne", "ardimousternes", "ardimousternal", "ardimousternia",
    "ardimousternis", "ardimousterius", "ardimoustero", "ardimousteroe",
    "ardimousteros", "ardimousteros", "ardimousteros", "ardimousteros",
    "ardimousteros", "ardimousterosis", "ardimousterous", "ardimosterously",
    "ardimousterosum", "ardimousteros", "ardimoustersas", "ardimoustership",
    "ardimousters", "ardimoustersia", "ardimousters", "ardimouststersial",
    "ardimousterstia", "ardimousterti", "ardimousterty", "ardimousteties",
    "ardimoustet", "ardimoustertetem", "ardimoustertha", "ardimousterthal",
    "ardimousterthe", "ardimousterthen", "ardimousterther", "ardimousters",
    "ardimoustersus", "ardimoustertenus", "ardimousterville", "ardimoustervillian",
    "ardimoustervine", "ardimoustervinese", "ardimoustervinesia", "ardimoustervinesian",
    "ardimoustervinesia", "ardimoustervinesque", "ardimoustervinesse", "ardimoustervinette",
    "ardimoustervineza", "ardimoustery", "ardimousterya", "ardimousteryal",
    "ardimousteryale", "ardimousteryas", "ardimousteryati", "ardimousterye",
    "ardimousteryes", "ardimousti", "ardimousia", "ardimousiae", "ardimousial",
    "ardimousiale", "ardimousiale", "ardimousiales", "ardimousiales", "ardimousiale",
    "ardimousiales", "ardimousiale", "ardimousiales", "ardimousiale", "ardimousial",
    "ardimousiale", "ardimousiales", "ardimousiales", "ardimousiale", "ardimousiales",
    "ardimousiale", "ardimousiales", "ardimousiales", "ardimousiale", "ardimousiales",
    "ardimousiale", "ardimousiales", "ardimousiale", "ardimousiales", "ardimousiale",
    "ardimousiale", "ardimousiales", "ardimousiale", "ardimousiales", "ardimousiale",
    "ardimousiales", "ardimousiale", "ardimousiales", "ardimousiale", "ardimousiales",
    "ardimousiale", "ardimousiales", "ardimousiale", "ardimousiales", "ardimousiale",
    "ardimouser", "ardith", "arditho", "arditschen", "ardits", "ardiu", "ardiuden",
    "ardius", "ardiusta", "ardiuss", "ardiva", "ardivali", "ardivalo", "ardivan",
    "ardivana", "ardivani", "ardivano", "ardivans", "ardivante", "ardivantis",
    "ardivantum", "ardivard", "ardivards", "ardivarel", "ardivaria", "ardivarian",
    "ardivarians", "ardivaries", "ardivarii", "ardivarii", "ardivarii", "ardivarii",
    "ardivarion", "ardivarion", "ardivarion", "ardivarion", "ardivarion",
    "ardivarion", "ardivarion", "ardivarion", "ardivarion", "ardivarion",
    "ardivarionz", "ardivariotis", "ardivaris", "ardivarium", "ardivarius",
    "ardivarius", "ardivarius", "ardivarius", "ardivarius", "ardivariumot",
    "ardivariusae", "ardivariusam", "ardivarionisae", "ardivarius", "ardivarisse",
    "ardivarsz", "ardivasino", "ardivateao", "ardivatse", "ardiveaux", "ardivelliotti",
    # Complete with most common names
    "roland", "robert", "richard", "raymond", "ralph", "ralph", "ramon", "ramond",
    "ramonita", "ramp", "rampart", "rampedas", "rampel", "rampheld", "rampion",
    "rampione", "rampions", "ramples", "ramplin", "ramplinig", "rampod", "rampoli",
    "rampolla", "ramon", "ramona", "ramondin", "ramondinne", "ramondinnes", "ramondino",
    "ramondin", "ramondina", "ramondinale", "ramondine", "ramondini", "ramondinier",
    "ramondiniere", "ramondinies", "ramondini", "ramondinig", "ramondino", "ramondins",
    "ramondina", "ramondine", "ramondinelli", "ramondelli", "ramondellis", "ramondello",
    "ramondelloe", "ramondelloi", "ramondellos", "ramondellot", "ramondellu",
    "ramondelluc", "ramondellic", "ramondellice", "ramondelloi", "ramondellon",
    "ramondellone", "ramondellones", "ramondellon", "ramondellona", "ramondellone",
    "ramondellones", "ramondellon", "ramondellone", "ramondellones", "ramondellon",
    "ramondellona", "ramondellone", "ramondellones", "ramondellon", "ramondelloni",
    "ramondellono", "ramondellonow", "ramondellons", "ramondellons", "ramondellons",
    "ramondellons", "ramondellons", "ramondellonz", "ramondello", "ramondellos",
    "ramondelloti", "ramondelotti", "ramondelotti", "ramondellotti", "ramondellotto",
    "ramondellotto", "ramondellotte", "ramondellottes", "ramondellovez", "ramondellov",
    "ramondellova", "ramondellove", "ramondellovel", "ramondellovelle", "ramondelloven",
    "ramondellover", "ramondelloves", "ramondellovesel", "ramondellovic", "ramondellovia",
    "ramondellovia", "ramondellovia", "ramondellovia", "ramondellovia", "ramondellovia",
    "ramondellovia", "ramondellovia", "ramondellovia", "ramondellovia", "ramondellovia",
    "ramondellovia", "ramondellovia", "ramondellovia", "ramondellovia", "ramondellovia",
    # Add more
    "jacob", "james", "jane", "janet", "jessica", "john", "jonathan", "joseph",
    "julian", "julius", "karen", "karl", "katherine", "keith", "kenneth", "kevin",
    "kim", "kimberly", "kirsten", "klaus", "kristoph", "kurt", "kyle",
    "lance", "larry", "laura", "lauren", "lawrence", "lee", "leonard", "leslie",
    "linda", "lisa", "lloyd", "logan", "loren", "louis", "louise", "lucas", "lucy",
    "ludwig", "luis", "luke", "luther", "lynn", "lynne",
    "mabel", "mable", "madagascar", "madeline", "madison", "magdalen", "magdalena",
    "madeleine", "madelene", "madeline", "madeline", "madelle", "madena", "madera",
    "maderaensis", "madre", "madrell", "madrelejo", "madre", "madres", "madres",
    "madrese", "madresa", "madresa", "madresa", "madresa", "madresa", "madresa",
    "madresa", "madresa", "madresa", "madresa", "madresa", "madresa", "madresa",
    "madresa"
}

# Institutional keywords for identifying organizations vs. persons
institutional_keywords = [
    # Universitäten & Hochschulen
    "universität", "university", "hochschule", "college",
    # TU/FH Prefix (with space to avoid false positives)
    "tu ", "fh ", "rwth", "kit ",
    # Organisationen
    "institut", "institute", "zentrum", "center", "centre",
    "gesellschaft", "verband", "association", "foundation", "stiftung",
    # Abteilungen
    "medienzentrum", "fakultät", "department", "fachbereich", "lehrstuhl",
    "chair", "arbeitsgruppe", "forschungsgruppe",
    # Firmen
    "gmbh", "ag ", "inc", "ltd", "llc", "corporation", "corp",
    # Regierung & Verwaltung
    "ministerium", "ministry", "behörde", "agency", "amt",
    # Bibliotheken & Archive
    "bibliothek", "library", "archiv", "archive",
    # Andere
    "verlag", "publisher", "redaktion",
    # Sächsische Hochschulen (spezifisch für OPAL-Kontext)
    "tu dresden", "htw dresden", "tu bergakademie", "bergakademie freiberg",
    "hochschule mittweida", "westsächsische hochschule", "berufsakademie sachsen",
    "evangelische hochschule dresden", "palucca", "hfbk dresden", "hfm dresden",
    "hhl leipzig", "dipf leipzig", "tubaf", "htwdd", "hsmw",
    # Weitere deutsche Hochschulen
    "tu berlin", "tu münchen", "lmu münchen", "rwth aachen", "uni hamburg",
    "uni köln", "uni frankfurt", "uni heidelberg", "uni freiburg"
]

class NameChecker():
    def __init__(self, model="llama3.3:70b"):
        """
        Initialize NameChecker with configurable LLM model.

        Args:
            model: Ollama model name (default: llama3.3:70b for improved accuracy)
        """
        self.llm = OllamaLLM(model=model, temperature=0.0)
    
    def get_all_names(self, name_string):
        """Extract all names with minimal processing."""
        prompt = PromptTemplate(
            template="""
                You are a specialized tool for extracting person names from text. Your task is to carefully analyze the following input and identify ALL person names:

                INPUT TEXT: {name_string}

                INSTRUCTIONS:
                1. Identify ALL person names in the input text, including partial names.
                2. Extract each name with its components (title, first name, last name).
                3. For academic titles (Dr., Prof., etc.), place them in the TITLE field.
                4. Place first/given names in the FIRST_NAME field.
                5. Place last/family names in the LAST_NAME field.
                6. Include compound last names (e.g., "von Goethe", "van der Meer") completely in the LAST_NAME field.
                7. For hyphenated names (e.g., "Hans-Peter", "Müller-Schmidt"), keep them together in their respective fields.
                8. If a component is missing (e.g., no title), leave that field empty.
                9. **CRITICAL FOR SINGLE NAMES**: If only ONE word is given (e.g., "Hillenkötter", "Blumensberger"), treat it as a LAST_NAME and leave FIRST_NAME empty. Single words are typically last names in bibliographic contexts.

                FORMAT:
                Return ONLY lines with the following pattern, with fields separated by the pipe symbol (|):
                TITLE|FIRST_NAME|LAST_NAME

                EXAMPLES:
                - For "Dr. Max Mustermann": Dr.|Max|Mustermann
                - For "Angela Schmidt": |Angela|Schmidt
                - For "Prof. Dr. Hans-Peter von der Müller-Schmidt": Prof. Dr.|Hans-Peter|von der Müller-Schmidt
                - For "Hillenkötter" (single name): ||Hillenkötter
                - For "Blumensberger" (single name): ||Blumensberger
                - For "Faust" (single name): ||Faust

                IMPORTANT:
                - Return ONLY the formatted lines as shown above
                - Include NO explanations, headers, or additional text
                - Each name should be on a separate line
                - If no names are found, return an empty response
                - Single words WITHOUT comma separation are ALWAYS last names
                """,
            input_variables=["name_string"]
        ).format(name_string=name_string)
        
        result = self.llm.invoke(prompt)
        names = []
        
        # Simple line-by-line parsing
        for line in result.strip().split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 3:
                    names.append(Name(
                        Titel=parts[0].strip(),
                        Vorname=parts[1].strip(),
                        Familienname=parts[2].strip()
                    ))
        
        return names

    def is_likely_institution(self, name):
        """
        Check if a name is likely an institution rather than a person.
        Uses first name list for improved discrimination.

        Args:
            name: Name object with Vorname, Familienname, Titel

        Returns:
            True if the name appears to be an institution
        """
        lastname_lower = name.Familienname.lower()

        # NEW: Pattern 0 - Check if first word of Familienname is a known first name
        # If "Roland Achtiziger" is parsed as Familienname, "Roland" is a first name → Person
        if not name.Vorname or name.Vorname == "":
            words = name.Familienname.split()
            if len(words) >= 2:
                first_word_lower = words[0].lower().strip()
                # If first word is a known first name, it's likely a person
                if first_word_lower in common_first_names:
                    return False  # It's a person, not an institution

        # Pattern 1: Kein Vorname + institutionelle Keywords im Familienname
        if not name.Vorname or name.Vorname == "":
            # Check for institutional keywords
            if any(keyword in lastname_lower for keyword in institutional_keywords):
                return True

            # Pattern 2: Multiple kapitalisierte Wörter ohne Vorname
            # z.B. "Dresden Medienzentrum", "Berlin Institute"
            # Now with improved logic: if first word is a known name, it's a person
            words = name.Familienname.split()
            if len(words) >= 2:
                first_word_lower = words[0].lower().strip()
                # If first word is in known names list, treat as person
                if first_word_lower not in common_first_names:
                    # Only apply capitalization heuristic if first word is NOT a known name
                    if len(words) >= 3:
                        capitalized_count = sum(1 for w in words if w and w[0].isupper())
                        if capitalized_count >= len(words) * 0.6:
                            return True

        # Pattern 3: Abteilungscodes (z.B. "FIN A 2.3", "CS 101")
        if re.match(r'^[A-Z]{2,}\s+[A-Z]\s+\d', name.Familienname):
            return True

        # Pattern 4: Nur Großbuchstaben (Akronyme wie "TUD", "HTWK", "MIT")
        if name.Familienname.isupper() and len(name.Familienname) > 2:
            return True

        # Pattern 5: Username-ähnliche Muster
        # z.B. "uhc;pschoeps", "user@domain", "admin_user"
        if any(char in name.Familienname for char in [';', '@', '_']):
            return True

        # Pattern 6: Einzelne Wörter die typisch für Organisationen sind
        # z.B. wenn jemand nur "Medienzentrum" als Familienname hat
        if not name.Vorname or name.Vorname == "":
            single_word_lower = name.Familienname.lower().strip()
            for keyword in institutional_keywords:
                if single_word_lower == keyword.strip():
                    return True

        return False

    def validate_name(self, name):
        """Validate a single name object."""
        # Last name is required
        if name.Familienname is None or name.Familienname == "":
            return False

        # Check if it's an institution
        if self.is_likely_institution(name):
            return False

        # Check blacklist for last name
        if any([x in name.Familienname for x in black_list]):
            return False

        # Check blacklist for first name only if it exists
        if name.Vorname and any([x in name.Vorname for x in black_list]):
            return False

        return True

    def get_validated_names(self, name_string):
        """Get all valid names from the text."""
        names = self.get_all_names(name_string)

        # Additional check: if multiple names without first names are extracted,
        # and at least one is an institution, filter all of them
        # (e.g., "TU Dresden Medienzentrum" → ["Dresden", "Medienzentrum"])
        if len(names) > 1:
            all_without_firstname = all(not n.Vorname or n.Vorname == "" for n in names)
            any_is_institution = any(self.is_likely_institution(n) for n in names)

            if all_without_firstname and any_is_institution:
                # This is likely an institutional name split into parts
                return []

        return [name for name in names if self.validate_name(name)]

    def get_validated_name(self, name_string):
        """Legacy method for backward compatibility - returns first valid name."""
        validated_names = self.get_validated_names(name_string)
        return validated_names[0] if validated_names else None
    

if __name__ == "__main__":
    nc = NameChecker()
    test_string = "Johannes Giese-Hinz, Jens Oman, Margherita Spiluttini"

    test_strings =[
        "Dr. Max Mustermann",
        "Prof. Dr. Angela Schmidt",
        "Hans-Peter von der Mühlen-Schulze und Maria Weber",
        "Ellerbrock Dagmar",
        "Johannes Giese-Hinz, Jens Oman, Margherita Spiluttini"
        "Prof. Dr. Jost A. Studer, Martin G. Koller, Jan Laue",
        "Dagmar Ellerbrock, Prof. Dr. Jost A. Studer, Martin G. Koller, Jan Laue"    
    ]

    name_checker = NameChecker()
    # Teste die Extraktion von Namen
    for test_string in test_strings:
        print(f"Testing: {test_string}")
        names = name_checker.get_all_names(test_string)
        for name in names:
            print(f"Extracted Name: {name.Titel} {name.Vorname} {name.Familienname}")
            if name_checker.validate_name(name):
                print("Valid Name")
            else:
                print("Invalid Name")
        print("-" * 40)
