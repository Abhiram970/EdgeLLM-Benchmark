# -*- coding: utf-8 -*-
"""
EdgeLM -- human-authored benchmark stimuli (all six task families).

CONTEXT.md standing rule (§3): ALL stimuli (passages, questions, conversation
scripts, prompts) are human-authored, sourced from existing human-written
material -- not model-generated. This file is the single auditable home for that
content so the contamination defense is inspectable in one place.

Provenance:
  * Tasks 1-3 passages: verbatim excerpts from Encyclopaedia Britannica
    (https://www.britannica.com), retrieved 2026-06-29. Each item records its
    source URL.
  * Task 4 (structured output) prompts: format-compliance instructions authored
    by a human; scoring is fully programmatic.
  * Task 5 (creative) prompts: human-authored constraint specs; quality is judged
    separately (small open judges unreliable -- CONTEXT.md §5).
  * Task 6 (code) problems: canonical algorithm/CS exercises (the kind found in
    any textbook) with human-authored I/O contracts and deterministic unit tests.
    The MODEL writes the solution; only the spec + tests are authored here, the
    same category of scaffolding as the QA gold answers.

Questions/gold answers (Tasks 1-3) are written to be answerable FROM THE PASSAGE
ALONE (answer-determinability, CONTEXT.md §3). Do NOT have an LLM rewrite or
"improve" any passage. An LLM judge may READ these when scoring; it must never
author or alter a stimulus.

Each task exposes a module-level list/dict the pipeline imports directly.
"""

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 -- CROSS-DOMAIN SUMMARIZATION  (6 passages: 2 prose, 2 math, 2 science)
# Headline signal = per-domain faithfulness + prose -> math/science drop. Passages
# are matched in length (~150-260 words). `key_claims` is a programmatic claim-
# survival check (each claim = list of synonym substrings; survives if ANY appears,
# case-insensitive). `reference_summary` is a human 3-5 sentence gold for ROUGE.
# `domain` is the scoring bucket: prose | mathematical | scientific.
# ─────────────────────────────────────────────────────────────────────────────

SUMMARIZATION_PASSAGES = [
    {
        "id": "sum_prose_frenchrev",
        "domain": "prose",
        "source": "https://www.britannica.com/event/French-Revolution",
        "text": (
            "The French Revolution had general causes common to all the revolutions of the West "
            "at the end of the 18th century and particular causes that explain why it was by far "
            "the most violent and the most universally significant of these revolutions. The first "
            "of the general causes was the social structure of the West. The feudal regime had been "
            "in steady decline since the Middle Ages and had already disappeared in parts of Europe. "
            "The increasingly numerous and prosperous elite of wealthy commoners—merchants, "
            "manufacturers, and professionals, often called the bourgeoisie—aspired to political "
            "power in those countries where it did not already possess it. Some of the peasants, many "
            "of whom owned land, had attained an improved standard of living and education and wanted "
            "to get rid of the last vestiges of feudalism so as to acquire the full rights of "
            "landowners and to be free to increase their holdings. About half of the peasants, "
            "however, remained poor or landless. Furthermore, from about 1730, higher standards of "
            "living had reduced the mortality rate among adults considerably. This, together with "
            "other factors, had led to an increase in the population of Europe unprecedented for "
            "several centuries: it doubled between 1715 and 1800. For France, which, with 26 million "
            "inhabitants in 1789, was the most populated country of Europe, the problem was most acute."
        ),
        "reference_summary": (
            "The French Revolution had both general causes shared with other late-18th-century "
            "Western revolutions and particular causes that made it the most violent and significant. "
            "A decaying feudal social structure left a prosperous bourgeoisie of merchants and "
            "professionals seeking political power they did not hold. Many landowning peasants wanted "
            "to shed the last of feudalism, while about half remained poor or landless. Rising living "
            "standards cut adult mortality and doubled Europe's population between 1715 and 1800, with "
            "France—the continent's most populous country at 26 million in 1789—feeling the strain most acutely."
        ),
        "key_claims": [
            ["bourgeoisie", "merchant", "wealthy commoner"],
            ["feudal", "feudalism"],
            ["peasant"],
            ["population", "mortality"],
            ["france", "26 million", "most popul"],
        ],
    },
    {
        "id": "sum_prose_industrial",
        "domain": "prose",
        "source": "https://www.britannica.com/event/Industrial-Revolution",
        "text": (
            "The Industrial Revolution increased the overall amount of wealth and distributed it more "
            "widely than had been the case in earlier centuries, helping to enlarge the middle class. "
            "However, the replacement of the domestic system of industrial production, in which "
            "independent craftspersons worked in or near their homes, with the factory system and mass "
            "production consigned large numbers of people, including women and children, to long hours "
            "of tedious and often dangerous work at subsistence wages. Their miserable conditions gave "
            "rise to the trade union movement in the mid-19th century. The Industrial Revolution "
            "transformed economies that had been based on agriculture and handicrafts into economies "
            "based on large-scale industry, mechanized manufacturing, and the factory system. New "
            "machines, new power sources, and new ways of organizing work made existing industries "
            "more productive and efficient. New industries also arose, including, in the late 19th "
            "century, the automobile industry. Historians conventionally divide the Industrial "
            "Revolution into two approximately consecutive parts. What is called the first Industrial "
            "Revolution lasted from the mid-18th century to about 1830 and was mostly confined to "
            "Britain. The second Industrial Revolution lasted from the mid-19th century until the "
            "early 20th century and took place in Britain, continental Europe, North America, and Japan."
        ),
        "reference_summary": (
            "The Industrial Revolution increased and more widely distributed wealth, enlarging the "
            "middle class, but it also forced many people—including women and children—into long, "
            "dangerous factory work at subsistence wages, which spurred the mid-19th-century trade "
            "union movement. It shifted economies from agriculture and handicrafts to large-scale "
            "mechanized industry and the factory system, raising productivity and creating new "
            "industries such as automobiles. Historians split it into a first phase (mid-18th century "
            "to about 1830, mostly in Britain) and a second phase (mid-19th to early 20th century) "
            "across Britain, Europe, North America, and Japan."
        ),
        "key_claims": [
            ["middle class", "wealth"],
            ["factory system", "factory"],
            ["trade union"],
            ["women and children", "women", "children"],
            ["first industrial", "second industrial", "two"],
        ],
    },
    {
        "id": "sum_math_calculus",
        "domain": "mathematical",
        "source": "https://www.britannica.com/science/calculus-mathematics",
        "text": (
            "The other great discovery of Newton and Leibniz was that finding the derivatives of "
            "functions was, in a precise sense, the inverse of the problem of finding areas under "
            "curves—a principle now known as the fundamental theorem of calculus. Specifically, "
            "Newton discovered that if there exists a function F(t) that denotes the area under the "
            "curve y = f(x) from, say, 0 to t, then this function’s derivative will equal the "
            "original curve over that interval, F′(t) = f(t). Hence, to find the area under the "
            "curve y = x2 from 0 to t, it is enough to find a function F so that F′(t) = t2. The "
            "differential calculus shows that the most general such function is x3/3 + C, where C is "
            "an arbitrary constant. This is called the (indefinite) integral of the function y = x2, "
            "and it is written as ∫x2dx. The initial symbol ∫ is an elongated S, which stands "
            "for sum, and dx indicates an infinitely small increment of the variable, or axis, over "
            "which the function is being summed. Leibniz introduced this because he thought of "
            "integration as finding the area under a curve by a summation of the areas of infinitely "
            "many infinitesimally thin rectangles between the x-axis and the curve. Newton and Leibniz "
            "discovered that integrating f(x) is equivalent to solving a differential equation—"
            "i.e., finding a function F(t) so that F′(t) = f(t)."
        ),
        "reference_summary": (
            "Newton and Leibniz discovered the fundamental theorem of calculus: differentiation and "
            "finding areas under curves are inverse operations. If F(t) gives the area under y = f(x) "
            "from 0 to t, then F'(t) = f(t), so computing an area reduces to finding an antiderivative. "
            "The indefinite integral, written with an elongated-S symbol standing for 'sum,' adds up "
            "infinitely many infinitesimally thin rectangles under the curve. Thus integrating a "
            "function is equivalent to solving the differential equation F'(t) = f(t)."
        ),
        "key_claims": [
            ["fundamental theorem"],
            ["newton", "leibniz"],
            ["inverse", "antiderivative", "indefinite integral"],
            ["area under", "area"],
            ["derivative", "differentiat"],
        ],
    },
    {
        "id": "sum_math_pythagorean",
        "domain": "mathematical",
        "source": "https://www.britannica.com/science/Pythagorean-theorem",
        "text": (
            "Pythagorean theorem, the well-known geometric theorem that the sum of the squares on the "
            "legs of a right triangle is equal to the square on the hypotenuse (the side opposite the "
            "right angle)—or, in familiar algebraic notation, a2 + b2 = c2. Although the theorem has "
            "long been associated with Greek mathematician-philosopher Pythagoras (c. 570–500/490 "
            "bce), it is actually far older. Four Babylonian tablets from circa 1900–1600 bce indicate "
            "some knowledge of the theorem, with a very accurate calculation of the square root of 2 "
            "(the length of the hypotenuse of a right triangle with the length of both legs equal to "
            "1) and lists of special integers known as Pythagorean triples that satisfy it (e.g., 3, "
            "4, and 5; 32 + 42 = 52, 9 + 16 = 25). The theorem is mentioned in the Baudhayana "
            "Sulba-sutra of India, which was written between 800 and 400 bce. Nevertheless, the "
            "theorem came to be credited to Pythagoras. It is also proposition number 47 from Book I "
            "of Euclid’s Elements. According to the Syrian historian Iamblichus, Pythagoras was "
            "introduced to mathematics by Thales of Miletus and his pupil Anaximander."
        ),
        "reference_summary": (
            "The Pythagorean theorem states that in a right triangle the sum of the squares on the two "
            "legs equals the square on the hypotenuse, written a² + b² = c². Although named for the "
            "Greek philosopher Pythagoras, it is far older: Babylonian tablets from about 1900–1600 "
            "BCE show knowledge of it, including Pythagorean triples such as 3, 4, 5, and it also "
            "appears in India's Baudhayana Sulba-sutra. It is recorded as proposition 47 in Book I of "
            "Euclid's Elements, yet it remained credited to Pythagoras."
        ),
        "key_claims": [
            ["right triangle"],
            ["hypotenuse"],
            ["a2 + b2", "a² + b²", "squares"],
            ["pythagoras"],
            ["babylonian", "older", "euclid", "triples"],
        ],
    },
    {
        "id": "sum_sci_tectonics",
        "domain": "scientific",
        "source": "https://www.britannica.com/science/plate-tectonics",
        "text": (
            "The concept of plate tectonics was formulated in the 1960s. According to the theory, "
            "Earth has a rigid outer layer, known as the lithosphere, which is typically about 100 km "
            "(60 miles) thick and overlies a plastic (moldable, partially molten) layer called the "
            "asthenosphere. The lithosphere is broken up into seven very large continental- and "
            "ocean-sized plates, six or seven medium-sized regional plates, and several small ones. "
            "These plates move relative to each other, typically at rates of 5 to 10 cm (2 to 4 "
            "inches) per year, and interact along their boundaries, where they converge, diverge, or "
            "slip past one another. Such interactions are thought to be responsible for most of "
            "Earth’s seismic and volcanic activity, although earthquakes and volcanoes can occur "
            "in plate interiors. Plate motions cause mountains to rise where plates push together, or "
            "converge, and continents to fracture and oceans to form where plates pull apart, or "
            "diverge. The continents are embedded in the plates and drift passively with them, which "
            "over millions of years results in significant changes in Earth’s geography."
        ),
        "reference_summary": (
            "Plate tectonics, formulated in the 1960s, holds that Earth's rigid ~100-km-thick "
            "lithosphere overlies the plastic, partially molten asthenosphere. The lithosphere is "
            "broken into about seven large plates plus medium and small ones that move a few "
            "centimeters per year. Where plates meet they converge, diverge, or slip past one "
            "another, producing most of Earth's earthquakes and volcanoes. Plate motion raises "
            "mountains where plates collide and opens oceans where they separate, carrying the "
            "embedded continents and reshaping Earth's geography over millions of years."
        ),
        "key_claims": [
            ["lithosphere"],
            ["asthenosphere"],
            ["plate", "plates"],
            ["converge", "diverge", "boundaries", "slip past"],
            ["earthquake", "volcan", "seismic"],
        ],
    },
    {
        "id": "sum_sci_dna",
        "domain": "scientific",
        "source": "https://www.britannica.com/science/DNA",
        "text": (
            "Each strand of a DNA molecule is composed of a long chain of monomer nucleotides. The "
            "nucleotides of DNA consist of a deoxyribose sugar molecule to which is attached a "
            "phosphate group and one of four nitrogenous bases: two purines (adenine and guanine) and "
            "two pyrimidines (cytosine and thymine). The nucleotides are joined together by covalent "
            "bonds between the phosphate of one nucleotide and the sugar of the next, forming a "
            "phosphate-sugar backbone linked by phosphodiester bonds, from which the nitrogenous bases "
            "protrude. One strand is held to another by hydrogen bonds between the bases; the "
            "sequencing of this bonding is specific—i.e., adenine bonds only with thymine, and "
            "cytosine only with guanine. In 1953 James Watson and Francis Crick, aided by the work of "
            "biophysicists Rosalind Franklin and Maurice Wilkins, determined that the structure of DNA "
            "is a double-helix polymer, a spiral consisting of two DNA strands wound around each other "
            "in opposite (antiparallel) directions. The breakthrough led to significant advances in "
            "scientists’ understanding of DNA replication and hereditary control of cellular activities."
        ),
        "reference_summary": (
            "Each DNA strand is a long chain of nucleotides, each made of a deoxyribose sugar, a "
            "phosphate group, and one of four nitrogenous bases: the purines adenine and guanine and "
            "the pyrimidines cytosine and thymine. Covalent phosphodiester bonds link the nucleotides "
            "into a sugar-phosphate backbone, and hydrogen bonds join two strands with specific "
            "pairing—adenine to thymine and cytosine to guanine. In 1953 Watson and Crick, aided by "
            "Franklin and Wilkins, showed DNA is an antiparallel double helix, advancing understanding "
            "of replication and heredity."
        ),
        "key_claims": [
            ["nucleotide"],
            ["adenine", "guanine", "cytosine", "thymine", "bases"],
            ["double-helix", "double helix"],
            ["watson", "crick"],
            ["hydrogen bond", "phosphate", "backbone"],
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 -- OPEN-BOOK COMPREHENSION  (6 passages x 3 graded questions)
# Domains: literary / scientific / historical (2 each). Tiers: literal ->
# inferential -> synthesis (easy/medium/hard). EVERY answer is determinable from
# the passage alone. `gold` = human reference; `accept` = acceptable substrings
# (any present => correct). `determinable=True` documents human verification.
# ─────────────────────────────────────────────────────────────────────────────

OPEN_BOOK_ITEMS = [
    {
        "id": "qa_lit_pride",
        "domain": "literary",
        "source": "https://www.britannica.com/topic/Pride-and-Prejudice",
        "passage": (
            "Pride and Prejudice is set in rural England in Hertfordshire and Derbyshire at the turn "
            "of the 19th century, and it centers on the Bennet family, which includes five very "
            "different sisters. The eldest, Jane, is sweet-tempered and modest. She is her sister "
            "Elizabeth’s confidant and friend. Elizabeth, or “Lizzy,” the heroine of "
            "the novel, is intelligent and high-spirited. She shares her father’s distaste for "
            "the conventional views of society as to the importance of wealth and rank. The third "
            "daughter, Mary, is plain, bookish, and pompous, while Lydia and Kitty, the two youngest, "
            "are flighty and immature."
        ),
        "questions": [
            {"difficulty": "easy", "q": "According to the passage, what two words describe the personalities of Lydia and Kitty?",
             "gold": "Flighty and immature.", "accept": ["flighty", "immature"], "determinable": True},
            {"difficulty": "medium", "q": "Who is the heroine of the novel, and what is her nickname?",
             "gold": "Elizabeth Bennet, nicknamed “Lizzy.”", "accept": ["elizabeth", "lizzy"], "determinable": True},
            {"difficulty": "hard",
             "q": "Based on the passage, which trait does Elizabeth share with her father, and what does that imply about her view of society?",
             "gold": "She shares her father's distaste for conventional views about the importance of wealth and rank, implying she does not judge people by social status.",
             "accept": ["wealth and rank", "conventional views", "distaste for", "social status"], "determinable": True},
        ],
    },
    {
        "id": "qa_lit_gatsby",
        "domain": "literary",
        "source": "https://www.britannica.com/topic/The-Great-Gatsby",
        "passage": (
            "F. Scott Fitzgerald’s novel The Great Gatsby (1925) contrasts the fictional East "
            "Coast village of West Egg, which represents “new money” with its brashness and "
            "materialism, and East Egg, which represents “old money” with its refinement and "
            "inherited wealth. The “old money” society ultimately triumphs as self-made "
            "millionaire Jay Gatsby’s pursuit of the American Dream ends in disaster. In the "
            "novel the green light at the end of Daisy Buchanan’s dock symbolizes Gatsby’s "
            "unattainable dream and ambition. A self-made millionaire who pursued Daisy in their youth "
            "in Kentucky, Gatsby renews his love affair with her in New York despite her marriage to "
            "wealthy Chicagoan Tom Buchanan."
        ),
        "questions": [
            {"difficulty": "easy", "q": "Where did Gatsby and Daisy originally meet in their youth?",
             "gold": "Kentucky.", "accept": ["kentucky"], "determinable": True},
            {"difficulty": "medium", "q": "What does the green light at the end of Daisy's dock symbolize?",
             "gold": "Gatsby's unattainable dream and ambition.", "accept": ["unattainable", "dream", "ambition"], "determinable": True},
            {"difficulty": "hard",
             "q": "According to the passage, what is the difference between West Egg and East Egg, and which society ultimately triumphs?",
             "gold": "West Egg represents brash, materialistic 'new money' and East Egg refined, inherited 'old money'; the old-money society ultimately triumphs.",
             "accept": ["old money", "old-money"], "determinable": True},
        ],
    },
    {
        "id": "qa_sci_blackhole",
        "domain": "scientific",
        "source": "https://www.britannica.com/science/black-hole",
        "passage": (
            "Details of the structure of a black hole are calculated from Albert Einstein’s "
            "general theory of relativity. The singularity constitutes the centre of a black hole and "
            "is hidden by the object’s “surface,” the event horizon. Inside the event "
            "horizon the escape velocity (i.e., the velocity required for matter to escape from the "
            "gravitational field of a cosmic object) exceeds the speed of light, so that not even rays "
            "of light can escape into space. The radius of the event horizon is called the "
            "Schwarzschild radius, after the German astronomer Karl Schwarzschild, who in 1916 "
            "predicted the existence of collapsed stellar bodies that emit no radiation. The size of "
            "the Schwarzschild radius is proportional to the mass of the collapsing star. For a black "
            "hole with a mass 10 times as great as that of the Sun, the radius would be 30 km "
            "(18.6 miles)."
        ),
        "questions": [
            {"difficulty": "easy", "q": "According to the passage, what is the Schwarzschild radius of a black hole with a mass 10 times that of the Sun?",
             "gold": "30 km.", "accept": ["30 km", "30km", "30 kilometers", "30 kilometres"], "determinable": True},
            {"difficulty": "medium", "q": "According to the passage, why can not even light escape from inside the event horizon?",
             "gold": "Because inside the event horizon the escape velocity exceeds the speed of light.",
             "accept": ["escape velocity", "exceeds the speed of light", "faster than light", "greater than the speed of light"], "determinable": True},
            {"difficulty": "hard",
             "q": "Using the passage, what would happen to the Schwarzschild radius if the mass of the collapsing star were doubled, and why?",
             "gold": "It would roughly double, because the Schwarzschild radius is proportional to the mass of the collapsing star.",
             "accept": ["would double", "also double", "roughly double", "proportional to the mass", "twice as"], "determinable": True},
        ],
    },
    {
        "id": "qa_sci_evolution",
        "domain": "scientific",
        "source": "https://www.britannica.com/science/evolution-scientific-theory",
        "passage": (
            "Natural selection is the process by which organisms adapt to their environment through "
            "the selective reproduction of advantageous genetic traits. Variations that enhance an "
            "organism's survival and ability to reproduce are preserved and passed on to future "
            "generations, while less advantageous variations diminish. This differential reproduction "
            "can occur due to differences in survival rates, fertility, mating success, or other life "
            "cycle aspects. Mutations are permanent alterations in an organism's genetic material that "
            "introduce new genetic variations; these variations are the raw material for evolution. "
            "While many mutations are neutral or harmful, some can be beneficial, improving an "
            "organism's ability to survive and reproduce. Natural selection then favors these "
            "advantageous mutations, causing them to increase in frequency over generations."
        ),
        "questions": [
            {"difficulty": "easy", "q": "According to the passage, through what four factors can differential reproduction occur?",
             "gold": "Survival rates, fertility, mating success, or other life cycle aspects.", "accept": ["survival rates", "fertility", "mating success", "life cycle"], "determinable": True},
            {"difficulty": "medium", "q": "What happens to variations that enhance an organism's survival and reproduction?",
             "gold": "They are preserved and passed on to future generations.", "accept": ["preserved", "passed on", "passed down", "future generation"], "determinable": True},
            {"difficulty": "hard",
             "q": "Based on the passage, explain how a beneficial mutation becomes common in a population over time.",
             "gold": "A beneficial mutation improves survival and reproduction, so natural selection favors it, causing it to increase in frequency over generations.",
             "accept": ["natural selection", "increase in frequency", "natural selection favors", "selection favors"], "determinable": True},
        ],
    },
    {
        "id": "qa_hist_frenchrev",
        "domain": "historical",
        "source": "https://www.britannica.com/event/French-Revolution",
        "passage": (
            "It is uncertain, however, whether revolution would have come without the added presence "
            "of a political crisis. Faced with the heavy expenditure that the wars of the 18th century "
            "entailed, the rulers of Europe sought to raise money by taxing the nobles and clergy, who "
            "in most countries had hitherto been exempt. To justify this, the rulers likewise invoked "
            "the arguments of advanced thinkers by adopting the role of “enlightened despots.” "
            "This provoked reaction throughout Europe from the privileged bodies, diets, and estates. "
            "In North America this backlash contributed to the American Revolution, which began with "
            "the refusal to pay a tax imposed by the king of Great Britain. Monarchs tried to stop "
            "this reaction of the aristocracy, and both rulers and the privileged classes sought "
            "allies among the nonprivileged bourgeois and the peasants."
        ),
        "questions": [
            {"difficulty": "easy", "q": "What role did the rulers claim to justify taxing the privileged classes?",
             "gold": "Enlightened despots.",
             "accept": ["enlightened despot", "enlightened despots"], "determinable": True},
            {"difficulty": "medium", "q": "Which two groups had usually been exempt from taxation before the rulers tried to tax them?",
             "gold": "The nobles and the clergy.", "accept": ["nobles and", "clergy"], "determinable": True},
            {"difficulty": "hard",
             "q": "According to the passage, how was the American Revolution connected to the same backlash occurring in Europe?",
             "gold": "The same aristocratic reaction against taxation appeared in North America, where the backlash contributed to the American Revolution, which began with a refusal to pay a tax imposed by the king of Great Britain.",
             "accept": ["refusal to pay", "king of great britain", "refused to pay", "refusal to pay a tax"], "determinable": True},
        ],
    },
    {
        "id": "qa_hist_printing",
        "domain": "historical",
        "source": "https://www.britannica.com/topic/printing-press",
        "passage": (
            "The earliest mention of a mechanized printing press in Europe appears in a lawsuit in "
            "Strasbourg in 1439; it reveals construction of a press for Johannes Gutenberg and his "
            "associates. Gutenberg’s press and others of its era in Europe owed much to the "
            "medieval paper press, which was in turn modeled after the ancient wine-and-olive press "
            "of the Mediterranean area. A long handle was used to turn a heavy wooden screw, exerting "
            "downward pressure against the paper, which was laid over the type mounted on a wooden "
            "platen. Gutenberg used his press to print an edition of the Bible in 1455; this Bible is "
            "the first complete extant book in the West, and it is one of the earliest books printed "
            "from movable type. (Jikji, a book of the teachings of Buddhist priests, was printed by "
            "hand from movable type in Korea in 1377.) In its essentials, the wooden press used by "
            "Gutenberg reigned supreme for more than 300 years."
        ),
        "questions": [
            {"difficulty": "easy", "q": "What mechanism did Gutenberg's press use to exert pressure on the paper?",
             "gold": "A long handle turning a heavy wooden screw.", "accept": ["wooden screw", "long handle", "heavy screw", "handle to turn"], "determinable": True},
            {"difficulty": "medium", "q": "What ancient device was the medieval paper press modeled after?",
             "gold": "The ancient wine-and-olive press of the Mediterranean area.", "accept": ["wine-and-olive", "wine and olive"], "determinable": True},
            {"difficulty": "hard",
             "q": "Based on the passage, what evidence shows that movable-type printing existed outside Europe before Gutenberg's Bible?",
             "gold": "Jikji, a book of Buddhist teachings, was printed from movable type in Korea in 1377—before Gutenberg's 1455 Bible.",
             "accept": ["jikji", "1377", "in korea", "korea in 1377"], "determinable": True},
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 -- CONTEXT RETENTION UNDER LOAD  (5 scripts, increasing turn depth)
# Each script plants tracked facts, interleaves human-authored (verbatim
# Britannica) distractor prose, and probes recall at growing depth. Probes score
# by exact-match recall. The pipeline logs token/context counts to separate
# window overflow from coherence collapse (CONTEXT.md §1).
#
# Step types: {"type":"plant","content":...}, {"type":"distractor","content":...},
# {"type":"probe","ask":...,"expect":[(label,[accept...]), ...]}.
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_RETENTION_SCRIPTS = [
    {
        "id": "ret_hydrologist",
        "facts": {"name": "Mara Okonkwo", "job": "hydrologist", "city": "Lisbon",
                  "pet": "Pim", "pet_kind": "tortoise", "pet_age": "12", "project_code": "DELTA-7"},
        "script": [
            {"type": "plant", "content": "My name is Mara Okonkwo and I work as a hydrologist in Lisbon."},
            {"type": "probe", "ask": "Just to confirm: what is my name and what is my job?",
             "expect": [("name", ["mara", "okonkwo"]), ("job", ["hydrolog"])]},
            {"type": "distractor", "content": (
                "Tell me if this is interesting: \"The concept of plate tectonics was formulated in "
                "the 1960s. Earth has a rigid outer layer, the lithosphere, about 100 km thick, "
                "overlying a plastic layer called the asthenosphere. The plates move at 5 to 10 cm "
                "per year.\"")},
            {"type": "plant", "content": "I have a tortoise named Pim who is 12 years old."},
            {"type": "probe", "ask": "Quick check: what is my name, my job, and the name of my pet?",
             "expect": [("name", ["mara", "okonkwo"]), ("job", ["hydrolog"]), ("pet", ["pim"])]},
            {"type": "distractor", "content": (
                "Another snippet: \"The singularity constitutes the centre of a black hole and is "
                "hidden by the event horizon. Inside it the escape velocity exceeds the speed of "
                "light, so not even light can escape.\" Thoughts?")},
            {"type": "plant", "content": "Also, I live in Lisbon and my current project is code-named DELTA-7."},
            {"type": "distractor", "content": (
                "One more: \"The French Revolution had general causes common to all the revolutions "
                "of the West at the end of the 18th century. The feudal regime had been in steady "
                "decline since the Middle Ages.\" Dense, right?")},
            {"type": "probe", "ask": (
                "Ignoring everything about tectonics, black holes, and the French Revolution: what "
                "is my name, my job, my city, my pet's name and species and age, and my project "
                "code name?"),
             "expect": [("name", ["mara", "okonkwo"]), ("job", ["hydrolog"]), ("city", ["lisbon"]),
                        ("pet", ["pim"]), ("pet_kind", ["tortoise"]), ("pet_age", ["12", "twelve"]),
                        ("project_code", ["delta-7", "delta 7", "delta7"])]},
        ],
    },
    {
        "id": "ret_chef",
        "facts": {"name": "Tomas Reyes", "job": "pastry chef", "city": "Montreal",
                  "signature": "lavender macaron", "opens_year": "2027"},
        "script": [
            {"type": "plant", "content": "Hi, I'm Tomas Reyes, a pastry chef based in Montreal."},
            {"type": "probe", "ask": "What's my name and my profession?",
             "expect": [("name", ["tomas", "reyes"]), ("job", ["pastry", "chef"])]},
            {"type": "distractor", "content": (
                "Random fact I read: \"Probabilities are numbers between 0 and 1, with 0 meaning "
                "impossible and 1 meaning certain. A probability of 0.5 means an event is equally "
                "likely to occur or not occur.\"")},
            {"type": "plant", "content": "My signature dessert is a lavender macaron, and I plan to open my own shop in 2027."},
            {"type": "distractor", "content": (
                "And this: \"The Pythagorean theorem states that in a right triangle the square of "
                "the hypotenuse equals the sum of the squares of the other two sides, a2 + b2 = "
                "c2.\"")},
            {"type": "probe", "ask": "Remind me: what is my name, my city, my signature dessert, and the year I plan to open my shop?",
             "expect": [("name", ["tomas", "reyes"]), ("city", ["montreal"]),
                        ("signature", ["lavender", "macaron"]), ("opens_year", ["2027"])]},
        ],
    },
    {
        "id": "ret_astronomer",
        "facts": {"name": "Yuki Tanaka", "job": "astronomer", "instrument": "radio telescope",
                  "target": "Andromeda", "grant_id": "NX-119"},
        "script": [
            {"type": "plant", "content": "I'm Yuki Tanaka, an astronomer. I mostly use a radio telescope."},
            {"type": "distractor", "content": (
                "Quick aside: \"DNA is made of nucleotides. A nucleotide has a sugar-phosphate "
                "backbone and one of four bases: adenine, guanine, cytosine, and thymine. Adenine "
                "bonds only with thymine.\"")},
            {"type": "plant", "content": "My main observation target is the Andromeda galaxy, funded under grant NX-119."},
            {"type": "distractor", "content": (
                "Also interesting: \"The Industrial Revolution transformed economies based on "
                "agriculture and handicrafts into economies based on large-scale industry and the "
                "factory system.\"")},
            {"type": "probe", "ask": "Recall for me: my name, my job, the instrument I use, my observation target, and my grant ID.",
             "expect": [("name", ["yuki", "tanaka"]), ("job", ["astronomer"]),
                        ("instrument", ["radio telescope", "radio"]), ("target", ["andromeda"]),
                        ("grant_id", ["nx-119", "nx 119", "nx119"])]},
        ],
    },
    {
        "id": "ret_archivist",
        "facts": {"name": "Priya Nair", "job": "archivist", "city": "Edinburgh",
                  "collection": "maritime maps", "oldest_year": "1602"},
        "script": [
            {"type": "plant", "content": "My name is Priya Nair and I'm an archivist in Edinburgh."},
            {"type": "probe", "ask": "Confirm my name and where I work.",
             "expect": [("name", ["priya", "nair"]), ("city", ["edinburgh"])]},
            {"type": "distractor", "content": (
                "Found this: \"The printing press first became mechanized in Europe. The earliest "
                "mention appears in a lawsuit in Strasbourg in 1439, revealing a press built for "
                "Johannes Gutenberg.\"")},
            {"type": "plant", "content": "I curate a collection of maritime maps; the oldest one dates to 1602."},
            {"type": "distractor", "content": (
                "And: \"Natural selection is the process by which organisms adapt to their "
                "environment through the selective reproduction of advantageous genetic traits.\"")},
            {"type": "probe", "ask": "What is my name, my job, my city, the collection I curate, and the year of the oldest item?",
             "expect": [("name", ["priya", "nair"]), ("job", ["archivist"]), ("city", ["edinburgh"]),
                        ("collection", ["maritime", "maps"]), ("oldest_year", ["1602"])]},
        ],
    },
    {
        "id": "ret_engineer",
        "facts": {"name": "Dario Conti", "job": "civil engineer", "city": "Oslo",
                  "structure": "suspension bridge", "span_m": "1400", "deadline": "March"},
        "script": [
            {"type": "plant", "content": "I'm Dario Conti, a civil engineer working in Oslo."},
            {"type": "distractor", "content": (
                "Read this earlier: \"The Roman Empire was founded in 27 BCE and in 395 AD split "
                "into Western and Eastern empires. The Western Roman Empire fell in 476.\"")},
            {"type": "plant", "content": "I'm designing a suspension bridge with a 1400-meter main span, due in March."},
            {"type": "distractor", "content": (
                "Also: \"A prime number greater than 1 can be expressed as a product of primes in a "
                "unique way; this is the fundamental theorem of arithmetic.\"")},
            {"type": "distractor", "content": (
                "And one more: \"Photosynthesis is the process by which green plants transform light "
                "energy into chemical energy, releasing oxygen as a by-product.\"")},
            {"type": "probe", "ask": "Tell me back: my name, my profession, my city, what I'm designing, its main span in meters, and the deadline month.",
             "expect": [("name", ["dario", "conti"]), ("job", ["civil engineer", "engineer"]),
                        ("city", ["oslo"]), ("structure", ["suspension bridge", "bridge"]),
                        ("span_m", ["1400", "1,400"]), ("deadline", ["march"])]},
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 -- STRUCTURED OUTPUT COMPLIANCE  (6 format tasks)
# Fully objective. `validator` names a function in the pipeline; `args` are passed
# to it. Each validator returns (strict, lenient, details). Strict = no prose
# preamble / no stray fences; lenient = the structure exists somewhere in output.
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_TASKS = [
    {"name": "json_person",
     "prompt": 'Return a JSON object with keys "name", "age", and "city" for a fictional person. Return ONLY valid JSON, no other text.',
     "validator": "json", "args": {"required": ["name", "age", "city"]}},
    {"name": "json_nested",
     "prompt": 'Return ONLY a JSON object with keys "title", "year", and "authors", where "authors" is a list of strings. No other text.',
     "validator": "json", "args": {"required": ["title", "year", "authors"]}},
    {"name": "markdown_doc",
     "prompt": "Write a short explanation of how photosynthesis works using proper Markdown formatting with at least one header, one bold term, and one bullet list.",
     "validator": "markdown", "args": {}},
    {"name": "code_only_fib",
     "prompt": "Write a Python function that returns the nth Fibonacci number. Return ONLY the code, no explanation, no markdown backticks.",
     "validator": "code_only", "args": {}},
    {"name": "numbered_list_planets",
     "prompt": "List exactly 5 planets in our solar system. Return ONLY a numbered list, nothing else.",
     "validator": "numbered_list", "args": {"expected_count": 5}},
    {"name": "csv_row",
     "prompt": "Output exactly one CSV line with three comma-separated values: a fruit, a color, and a price. Return ONLY the CSV line, no header, no other text.",
     "validator": "csv_line", "args": {"fields": 3}},
]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 -- CREATIVE GENERATION  (6 constrained pieces)
# Constraint satisfaction is objective (checked in the pipeline). Quality is left
# for a stronger judge / human slice (CONTEXT.md §5). Each item declares which
# objective checks apply.
# ─────────────────────────────────────────────────────────────────────────────

CREATIVE_TASKS = [
    {"type": "article",
     "prompt": "Write a 200-word news article about a fictional discovery of a new species of deep-sea fish. Include a headline.",
     "constraints": ["has headline", "~200 words", "journalistic tone"],
     "target_words": 200},
    {"type": "poem_unrhymed",
     "prompt": "Write a 12-line poem about the feeling of watching a city at night from a rooftop. Do not rhyme.",
     "constraints": ["12 lines", "no rhyme", "vivid imagery"],
     "target_lines": 12},
    {"type": "short_story",
     "prompt": "Write a 150-word short story that begins with the sentence: 'The last bus had already left.' Include a clear beginning, middle, and end.",
     "constraints": ["starts with given sentence", "~150 words", "complete arc"],
     "target_words": 150, "must_start": "the last bus had already left"},
    {"type": "acrostic",
     "prompt": "Write an acrostic poem where the first letters of the lines spell OCEAN. Exactly 5 lines.",
     "constraints": ["5 lines", "acrostic spells OCEAN"],
     "target_lines": 5, "acrostic": "OCEAN"},
    {"type": "dialogue",
     "prompt": "Write a 10-line dialogue between two characters arguing about whether to sell an old family house. Each line must start with the speaker's name followed by a colon. Exactly 10 lines.",
     "constraints": ["10 lines", "name: prefix each line"],
     "target_lines": 10, "line_prefix_colon": True},
    {"type": "six_word_story",
     "prompt": "Write a complete story in exactly six words. Return ONLY the six words.",
     "constraints": ["exactly 6 words"],
     "exact_words": 6},
]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 -- CODE GENERATION  (6 problems; Python executed, obscure langs static)
# Canonical CS exercises. The MODEL writes the solution; only the spec + tests are
# authored here. Python/Perl get real execution where the runtime is available;
# C# is static-checked. `test` (Python) = a harness fragment appended after the
# model's code that must print EDGELM_PASS. `static` (other langs) = substrings/
# predicate checked against the generated code.
# ─────────────────────────────────────────────────────────────────────────────

CODE_TASKS = [
    {
        "name": "py_process_csv",
        "language": "python",
        "prompt": (
            "Write a Python function called `process_csv` that:\n"
            "1. Takes a filename as input\n"
            "2. Reads a CSV with columns 'name', 'score', 'grade'\n"
            "3. Returns a dict with keys 'average_score' (float) and 'a_students' "
            "(list of names with grade 'A')\n"
            "Return ONLY the Python code, no explanation, no markdown."
        ),
        "entrypoint": "process_csv",
        "test": (
            "import csv as _csv, os as _os, tempfile as _tf\n"
            "_rows=[{'name':'Ana','score':'90','grade':'A'},{'name':'Ben','score':'70','grade':'C'},"
            "{'name':'Cleo','score':'95','grade':'A'}]\n"
            "_fd,_p=_tf.mkstemp(suffix='.csv')\n"
            "import os as _o\n"
            "with _o.fdopen(_fd,'w',newline='') as _f:\n"
            "    _w=_csv.DictWriter(_f,fieldnames=['name','score','grade']); _w.writeheader(); _w.writerows(_rows)\n"
            "try:\n"
            "    _r=process_csv(_p)\n"
            "    assert abs(float(_r['average_score'])-(90+70+95)/3)<0.5, ('avg',_r)\n"
            "    assert set(_r['a_students'])=={'Ana','Cleo'}, ('a',_r)\n"
            "    print('EDGELM_PASS')\n"
            "finally:\n"
            "    _o.unlink(_p)\n"
        ),
    },
    {
        "name": "py_is_palindrome",
        "language": "python",
        "prompt": (
            "Write a Python function `is_palindrome(s)` that returns True if the string s is a "
            "palindrome, ignoring case, spaces, and punctuation, and False otherwise. "
            "Return ONLY the Python code."
        ),
        "entrypoint": "is_palindrome",
        "test": (
            "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
            "assert is_palindrome('race a car') is False\n"
            "assert is_palindrome('') is True\n"
            "assert is_palindrome('No lemon, no melon') is True\n"
            "print('EDGELM_PASS')\n"
        ),
    },
    {
        "name": "py_fizzbuzz",
        "language": "python",
        "prompt": (
            "Write a Python function `fizzbuzz(n)` that returns a list of length n where the i-th "
            "element (1-indexed) is 'FizzBuzz' if i is divisible by 15, 'Fizz' if divisible by 3, "
            "'Buzz' if divisible by 5, otherwise the number i as a string. Return ONLY the code."
        ),
        "entrypoint": "fizzbuzz",
        "test": (
            "r=fizzbuzz(15)\n"
            "assert r[0]=='1' and r[2]=='Fizz' and r[4]=='Buzz' and r[14]=='FizzBuzz', r\n"
            "assert len(r)==15, len(r)\n"
            "print('EDGELM_PASS')\n"
        ),
    },
    {
        "name": "py_two_sum",
        "language": "python",
        "prompt": (
            "Write a Python function `two_sum(nums, target)` that returns a list of two indices "
            "[i, j] (i < j) such that nums[i] + nums[j] == target. Assume exactly one solution "
            "exists. Return ONLY the code."
        ),
        "entrypoint": "two_sum",
        "test": (
            "r=two_sum([2,7,11,15],9)\n"
            "assert sorted(r)==[0,1], r\n"
            "r2=two_sum([3,2,4],6)\n"
            "assert sorted(r2)==[1,2], r2\n"
            "print('EDGELM_PASS')\n"
        ),
    },
    {
        "name": "perl_error_counter",
        "language": "perl",
        "prompt": (
            "Write a Perl script that takes a filename as its first command-line argument "
            "($ARGV[0]), reads that text file line by line, counts the total number of lines, counts "
            "the number of lines containing the word 'error' (case-insensitive), and prints both "
            "counts (each count as a number). Return ONLY the Perl code, no explanation."
        ),
        # static check (Perl runtime may be absent on the box); also executed if `perl` exists
        "static": lambda code: (("while" in code.lower() or "foreach" in code.lower())
                                and "print" in code.lower()
                                and ("error" in code.lower())),
        "exec_perl": {
            # fixture lines + expected stdout substrings when run as: perl script.pl fixture.txt
            "fixture": "all good\nERROR here\nnothing\nminor error happened\nok\n",
            "expect_substrings": ["5", "2"],  # 5 total lines, 2 with 'error'
        },
    },
    {
        "name": "csharp_unity_controller",
        "language": "csharp",
        "prompt": (
            "Write a Unity C# MonoBehaviour script for a first-person player controller. Include: "
            "WASD movement, mouse look, jump with spacebar, and gravity. Return ONLY the C# code, "
            "no explanation."
        ),
        "static": lambda code: ("MonoBehaviour" in code and "Update" in code
                                and ("Input.GetAxis" in code or "Input.GetKey" in code)),
    },
]
