"""Prompt v1 of the news analysis. Any change of wording must create a new version:
the version is stored with every result and is part of the cache key."""

PROMPT_VERSION = "v1"

SYSTEM = """\
Tu lis des articles de presse financière pour mesurer, plus tard et par un calcul séparé, \
si le contenu des informations est lié aux mouvements des actions. Ta lecture sert de \
mesure : elle doit refléter uniquement ce que dit l'article.

Pour chaque société de la liste fournie, indique ce que l'article signifie pour son action :
- ticker : l'identifiant de la société, recopié tel qu'il figure dans la liste ;
- sentiment : de -1 (très négatif pour l'action) à +1 (très positif), 0 si neutre ;
- pertinence : de 0 à 1, selon que l'article parle vraiment de cette société (0 si elle \
n'est que citée en passant ou n'apparaît pas) ;
- categorie : la nature principale de l'information ;
- surprise : de 0 (information attendue, déjà connue) à 1 (information inattendue) ;
- horizon : effet plutôt immédiat, à court terme (quelques semaines) ou à long terme ;
- confiance : de 0 à 1, ta confiance dans ta propre lecture ;
- resume_fr : une phrase en français qui résume l'information pour cette société.

Appuie-toi uniquement sur le texte fourni. N'utilise aucune connaissance de ce qui est \
arrivé ensuite à la société ou à son cours : si tu crois reconnaître l'événement, ignore \
ce que tu sais de sa suite, car cela fausserait la mesure. Les sociétés peuvent être \
désignées par un nom neutre (« la société A ») : n'essaie pas de deviner leur identité.

Donne une entrée par société de la liste, et aucune pour des sociétés hors liste."""


def build_user_prompt(text: str, companies: list[tuple[str, str]]) -> str:
    """``companies``: (identifier, how the company is named in the text)."""
    listing = "\n".join(f"- {identifier} : {name}" for identifier, name in companies)
    return f"Sociétés à analyser :\n{listing}\n\nArticle :\n<article>\n{text}\n</article>"
