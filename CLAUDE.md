# CLAUDE.md — Magnificent Muge Trading

Simulateur de trading (backtest) + volet réel (plus tard).

## 0. Nom et identité visuelle

- **Nom de l'application : Magnificent Muge Trading** (nom court : *Muge Trading*). Le muge est le mulet de Méditerranée, clin d'œil marseillais.
- Nom du dépôt / dossier : `magnificent-muge-trading` ; nom du package Python : `mugetrading`.
- **Logo** : `assets/logo.png` (1254 × 1254 px, fond sombre). Icône d'onglet : `assets/favicon.png` (64 × 64 px).

Utilisation dans l'interface Streamlit :
- `st.set_page_config(page_title="Magnificent Muge Trading", page_icon="assets/favicon.png", layout="wide")` ;
- logo affiché en haut de la barre latérale (`st.logo` ou `st.sidebar.image`, largeur ~180 px) ;
- sur la page d'accueil : logo plus grand et centré, avec le nom en titre ;
- si `assets/logo.png` est absent, l'application démarre quand même (nom en texte à la place du logo) ;
- nom et petit logo repris en en-tête des exports (CSV : première ligne de commentaire avec le nom de l'appli et la date ; futurs exports PDF/PNG : logo en en-tête).

Thème visuel, aligné sur le logo — **thème sombre par défaut** (`.streamlit/config.toml`) :

| Rôle | Couleur |
|---|---|
| Fond | bleu nuit `#0B1220` |
| Fond secondaire (cartes, barre latérale) | `#131C2E` |
| Couleur principale (boutons, liens, courbe du portefeuille) | bleu électrique `#2EA8FF` |
| Accent (titres, indicateurs clés) | or `#D4A85A` |
| Texte | `#E8EDF5` |
| Hausse / achat | vert `#2ECC8F` |
| Baisse / vente | rouge `#E5484D` |

Les graphiques Plotly utilisent le même thème (fond transparent, grille discrète, vert/rouge ci-dessus pour achats/ventes et chandeliers).

## 1. Contexte

Application personnelle, utilisée par deux personnes (Gilles et un ami), toutes deux débutantes en bourse.
But : tester des stratégies simples d'achat/vente d'actions sur des données historiques réelles, avant d'envisager de petits ordres réels (capital de départ envisagé : **1 000 € maximum**).

L'application a deux volets :

1. **Volet Simulation (à développer maintenant)** : téléchargement de l'historique des cours, réglage des paramètres de stratégie, simulation des achats/ventes, restitution en tableau et en graphiques.
2. **Volet Réel (plus tard, NE PAS IMPLÉMENTER)** : passage d'ordres réels via un courtier. Seule une interface abstraite est prévue dans l'architecture pour ne pas avoir à tout refaire (voir §9).

> ⚠️ Règle absolue : aucun code ne doit envoyer d'ordre réel, ni se connecter à un compte courtier, tant que le volet Réel n'a pas été explicitement lancé.

---

## 2. Stack technique

- **Langage** : Python 3.11+
- **Données de marché** : source choisie par l'utilisateur dans l'onglet **Paramétrage** (voir §4 bis). **V1 : Yahoo Finance via `yfinance`** (gratuit, cours journaliers OHLCV), seule source implémentée au départ. Tout accès aux données passe par une interface `DataProvider`, pour pouvoir ajouter d'autres sources sans toucher au reste du code (yfinance n'est pas une API officielle et peut casser).
- **Calcul** : `pandas`, `numpy`
- **Stockage local** : fichiers Parquet (cache des cours) + SQLite (paramètres et résultats des simulations sauvegardées)
- **Interface** : `Streamlit` (application web locale, lancée avec `streamlit run app.py`)
- **Graphiques** : `plotly`
- **Tests** : `pytest`
- **Gestion des dépendances** : `pyproject.toml` + environnement virtuel (`uv` ou `venv`)

Contraintes d'environnement :
- Doit tourner sur macOS **Apple Silicon et Intel** (dont un vieux Mac sans AVX2) → éviter les dépendances lourdes ou compilées exotiques (pas de TA-Lib, pas de bibliothèque de backtest tierce). Le moteur de simulation est écrit à la main.
- Pas de service cloud, pas de compte à créer. Tout est local.

Conventions :
- Code, noms de variables et commentaires techniques en **anglais**.
- **Interface utilisateur entièrement en français** (libellés, tableaux, graphiques, messages d'erreur).
- Montants en euros, format français (`1 234,56 €`), dates `JJ/MM/AAAA` dans l'interface.

---

## 3. Arborescence cible

```
magnificent-muge-trading/
├── CLAUDE.md
├── pyproject.toml
├── app.py                     # point d'entrée Streamlit
├── .streamlit/
│   └── config.toml            # thème sombre aux couleurs du logo
├── assets/
│   ├── logo.png               # logo Magnificent Muge Trading
│   └── favicon.png            # icône d'onglet
├── config/
│   └── fee_profiles.yaml      # grilles de frais des plateformes (modifiable à la main)
├── src/mugetrading/
│   ├── data/
│   │   ├── provider.py        # interface DataProvider
│   │   ├── registry.py        # registre des sources disponibles
│   │   ├── yfinance_provider.py
│   │   └── cache.py           # lecture/écriture Parquet (un cache par source)
│   ├── strategy/
│   │   ├── params.py          # dataclass StrategyParams
│   │   ├── signals.py         # momentum, breakout
│   │   └── stops.py           # stop loss, stop suiveur
│   ├── engine/
│   │   ├── portfolio.py       # liquidités, positions, valorisation
│   │   ├── fees.py            # modèle de frais par tranches + chargement des profils
│   │   └── backtest.py        # boucle de simulation jour par jour
│   ├── broker/
│   │   ├── base.py            # interface Broker (abstraite)
│   │   └── paper.py           # implémentation simulée (utilisée par le backtest)
│   ├── storage/
│   │   └── db.py              # SQLite : simulations sauvegardées
│   └── ui/
│       ├── page_settings.py   # onglet Paramétrage (source de données, etc.)
│       ├── page_data.py       # écran téléchargement des données
│       ├── page_simulation.py # écran paramètres + lancement
│       └── page_results.py    # écran tableau + graphiques
├── data/                      # cache Parquet (ignoré par git)
└── tests/
```

---

## 4. Fonction 1 — Téléchargement des données

L'utilisateur choisit :
- **un univers d'actions** :
  - listes prédéfinies : CAC 40, SBF 120 (tickers Yahoo, ex. `AIR.PA`, `MC.PA`) ;
  - ou une liste personnalisée saisie à la main (tickers séparés par des virgules) ;
- **une date de début et une date de fin** ;
- **un indice de référence** pour comparaison (par défaut CAC 40 : `^FCHI`).

Le programme télécharge, pour chaque titre, les **cours journaliers** : date, ouverture, plus haut, plus bas, clôture, clôture ajustée, volume.

Exigences :
- Mise en cache locale : ne pas retélécharger ce qui est déjà en cache ; ne télécharger que les jours manquants.
- Télécharger automatiquement une marge **avant** la date de début (au moins la plus longue période de calcul des indicateurs, + 10 jours) pour que les indicateurs soient calculables dès le premier jour de la simulation.
- Utiliser les cours **ajustés** (dividendes et divisions d'actions) pour les calculs.
- Afficher une barre de progression, puis un récapitulatif : nombre de titres, nombre de jours, titres en échec ou avec trous de données.
- Un titre sans données sur une partie de la période est simplement ignoré sur cette partie (pas de plantage).

Granularité : **journalière uniquement** en V1. (L'intraday gratuit est trop limité en profondeur d'historique.)

Le téléchargement utilise la source sélectionnée dans l'onglet Paramétrage ; la source utilisée est rappelée en haut de l'écran Données.

---

## 4 bis. Onglet Paramétrage — source de données et valeurs par défaut

Un onglet **Paramétrage** permet de choisir la source de données de marché.

- Liste déroulante des sources, alimentée par `data/registry.py`.
- **V1 : seule « Yahoo Finance » est implémentée et sélectionnée par défaut.**
- Les sources prévues ensuite apparaissent dans la liste, **grisées avec la mention « Bientôt disponible »** :
  - EODHD (EOD Historical Data) : payant, inclut les sociétés retirées de la cote (utile pour éviter le biais des seuls survivants) ;
  - Interactive Brokers : à relier au futur volet Réel.
- Pour les sources qui en ont besoin, un champ **clé API** (saisie masquée). Les clés sont stockées dans un fichier `.env` local, **jamais en base ni dans git** (`.env` dans `.gitignore`, fournir un `.env.example`).
- Bouton **« Tester la connexion »** : télécharge quelques jours d'un titre témoin (ex. `MC.PA`) et affiche un message de succès ou d'erreur clair.
- Le choix est mémorisé (SQLite, table `settings`) et repris au prochain lancement.

L'onglet Paramétrage contient aussi les **valeurs par défaut** des nouvelles simulations, à commencer par le **capital de départ** (1 000 € par défaut, voir §5.3), ainsi que la **plateforme (profil de frais) par défaut** et l'édition des profils de frais (voir §5.4).

Règles techniques :
- Chaque source implémente `DataProvider` et renvoie exactement le même format (colonnes, index de dates, devise, cours ajustés) pour que le moteur ne voie aucune différence.
- Chaque source gère sa propre conversion de tickers (ex. `MC.PA` chez Yahoo ↔ `MC.PA` ou autre format ailleurs) : l'utilisateur saisit toujours le format Yahoo.
- Le cache Parquet est **séparé par source** (`data/<source>/…`) pour ne jamais mélanger des données d'origines différentes.
- Chaque simulation sauvegardée enregistre la source de données utilisée, affichée dans les résultats et dans la comparaison de simulations.

Ajouter une nouvelle source = écrire un module `xxx_provider.py` et l'inscrire dans le registre, rien d'autre.

---

## 5. Fonction 2 — Paramètres de la stratégie

Tous les paramètres sont réglables dans l'interface (curseurs / champs numériques), avec une valeur par défaut et une infobulle d'explication en français simple.

### 5.1 Signaux d'achat

| Paramètre | Définition | Défaut |
|---|---|---|
| **Momentum – période** | Nombre de jours de bourse pour mesurer la variation | 20 j |
| **Momentum – seuil** | Signal si `(clôture J / clôture J−période − 1) ≥ seuil` | +5 % |
| **Breakout – période** | Signal si la clôture J dépasse le **plus haut** des N jours précédents (J exclu) | 20 j |
| **Breakout – marge** | Dépassement minimum au-dessus de ce plus haut | 0 % |
| **Combinaison des signaux** | `Momentum seul`, `Breakout seul`, `Les deux (ET)`, `L'un ou l'autre (OU)` | ET |

Chaque signal doit pouvoir être activé/désactivé.

### 5.2 Signaux de vente

| Paramètre | Définition | Défaut |
|---|---|---|
| **Stop loss** | Vente si le cours tombe à `prix d'achat × (1 − x %)` | 8 % |
| **Stop suiveur** | Vente si le cours tombe à `plus haut atteint depuis l'achat × (1 − y %)` | 10 % |

- Chaque stop peut être activé/désactivé.
- Si les deux sont actifs, le niveau de stop retenu chaque jour est le **plus élevé** des deux.
- Le plus haut du stop suiveur est mis à jour avec le **plus haut du jour**, mais le nouveau niveau de stop ne s'applique qu'**à partir du lendemain** (pas d'utilisation du futur).

### 5.3 Gestion du portefeuille

| Paramètre | Définition | Défaut |
|---|---|---|
| **Capital de départ** | Liquidités au départ (voir précisions ci-dessous) | 1 000 € |
| Taille d'une position | % de la valeur du portefeuille investi par achat | 25 % |
| Nombre max de positions simultanées | | 4 |
| Fractions d'actions | Autoriser ou non l'achat de fractions (sinon nombre entier, arrondi à l'inférieur) | Non |
| Priorité si plusieurs signaux le même jour | Classer les candidats par momentum décroissant | — |

**Capital de départ — précisions :**
- Champ de saisie libre en euros, bien visible en tête de l'écran Simulation (pas un simple curseur), avec quelques raccourcis : 500 €, 1 000 €, 5 000 €, 10 000 €.
- Minimum 100 €, pas de maximum imposé ; valeur arrondie au centime.
- Une **valeur par défaut** se règle dans l'onglet Paramétrage (§4 bis) ; elle peut être modifiée pour chaque simulation.
- Le capital de départ est enregistré avec chaque simulation sauvegardée et affiché dans les résultats et la comparaison.
- L'indice de référence (§7.2) est toujours ramené au même capital de départ, pour une comparaison juste.
- Si le capital est faible par rapport aux frais minimum (ex. frais minimum > 1 % de la taille d'une position), afficher un avertissement : « Avec ce capital, les frais vont peser lourd sur chaque opération ».

Si les liquidités ne suffisent pas pour au moins 1 action + frais, l'achat est ignoré (et noté comme tel dans un journal des signaux non exécutés).

### 5.4 Frais de transaction

#### 5.4.1 Choix de la plateforme

Dans l'écran Simulation, une liste déroulante **« Plateforme (courtier) »** permet de choisir le courtier dont on veut reproduire les frais. Les frais de chaque opération du journal sont alors calculés selon la grille de ce courtier.

À côté de la liste, afficher un **aperçu** : coût d'un ordre de 250 €, 500 €, 1 000 € et 2 500 € avec la plateforme choisie.

Un bouton **« Comparer les plateformes »** relance la même simulation (mêmes données, mêmes paramètres) avec chaque plateforme et affiche un tableau : plateforme, total des frais payés, valeur finale du portefeuille, performance nette. C'est la même stratégie, seul le courtier change.

#### 5.4.2 Modèle de frais générique

Chaque plateforme est décrite par un **profil de frais** composé de **tranches** selon le montant de l'ordre. Pour chaque tranche :

```
frais_tranche = min(maximum, max(minimum, fixe + pourcentage × montant))
```

Champs d'un profil :
- `name`, `description`
- `brackets` : liste de tranches `{jusqu_a, fixe, pourcentage, minimum, maximum}` (la dernière tranche sans plafond de montant)
- `free_orders_per_month` + `free_orders_max_amount` : ordres gratuits par mois sous un montant donné (ex. Fortuneo)
- `account_type` : `CTO` ou `PEA`. En PEA, les frais sont plafonnés à **0,5 % du montant de l'ordre** (plafond légal des ordres passés en ligne).
- `verified_on` : date à laquelle la grille a été vérifiée, et `source_url` : lien vers la grille officielle

S'ajoutent, indépendamment de la plateforme :

| Paramètre | Défaut |
|---|---|
| Taxe sur les transactions financières (TTF, à l'achat, actions françaises concernées) | 0,4 %, désactivable |
| Glissement (slippage) : écart défavorable appliqué au prix d'exécution | 0,05 % |

#### 5.4.3 Profils fournis au départ

Les profils sont stockés dans un fichier **`config/fee_profiles.yaml`**, modifiable à la main sans toucher au code. Les grilles des courtiers changent souvent : chaque profil affiche sa date de vérification dans l'interface, avec un avertissement si elle date de plus de 6 mois.

Valeurs initiales (actions Euronext Paris, compte-titres, vérifiées en novembre 2026 sur des comparatifs ; **à revérifier sur les grilles officielles avant usage**) :

| Plateforme | Grille (ordre sur Euronext Paris) |
|---|---|
| **Trade Republic** | 1 € par ordre, quel que soit le montant |
| **DEGIRO** | 2 € par ordre (1 € de courtage + 1 € de frais de gestion), quel que soit le montant |
| **Bourse Direct** | ≤ 500 € : 0,99 € · 500–1 000 € : 1,90 € · 1 000–2 000 € : 2,90 € · 2 000–4 400 € : 3,80 € · > 4 400 € : 0,09 % |
| **BoursoBank – Découverte** | ≤ 500 € : 1,99 € · au-delà : **à compléter** depuis la grille officielle |
| **Fortuneo – Starter** | 1er ordre du mois ≤ 500 € gratuit, sinon 0,35 % du montant |
| **Interactive Brokers – Fixed** | Pourcentage du montant avec minimum 3 € par ordre : **pourcentage à compléter** depuis la grille officielle |
| **Personnalisé** | Saisie libre de la grille dans l'onglet Paramétrage |
| **Sans frais** | 0 € (pour isoler l'effet de la stratégie seule) |

Dans l'onglet Paramétrage :
- choix de la **plateforme par défaut** des nouvelles simulations ;
- édition d'un profil existant ou création d'un profil personnalisé (formulaire des tranches) ;
- les modifications sont réécrites dans `config/fee_profiles.yaml`.

Chaque simulation sauvegardée enregistre le profil de frais utilisé **tel qu'il était** au moment de la simulation (copie des valeurs, pas seulement le nom), pour que les résultats restent reproductibles si la grille est modifiée ensuite.

#### 5.4.4 Tests des frais

Tests `pytest` pour chaque profil fourni : vérifier le montant des frais pour des ordres de 100 €, 500 €, 501 €, 1 000 €, 4 400 € et 10 000 €, le compteur d'ordres gratuits mensuels (remis à zéro chaque mois) et le plafond PEA.

---

## 6. Fonction 3 — Moteur de simulation

Boucle **jour par jour** sur la période choisie. Règles impératives pour éviter les résultats trop beaux pour être vrais :

1. **Pas d'utilisation du futur** : un signal calculé sur la clôture du jour J est exécuté à **l'ouverture du jour J+1**.
2. **Stops** : vérifiés chaque jour sur le plus bas du jour.
   - Si le cours **ouvre déjà sous le stop** (trou de cotation) → vente au cours d'ouverture.
   - Sinon, si le plus bas du jour touche le stop → vente au niveau du stop.
3. Ordre de traitement dans une journée : d'abord les **ventes** (stops), puis les **achats** (signaux de la veille), avec les liquidités disponibles après les ventes.
4. Pas de rachat d'un titre le jour même où il a été vendu.
5. En fin de période, les positions encore ouvertes sont **valorisées** à la dernière clôture (option : les vendre fictivement, motif « Fin de période »).
6. Valorisation du portefeuille chaque jour à la clôture : `liquidités + Σ (quantité × clôture)`.

Le moteur est une fonction pure, sans dépendance à Streamlit :
`run_backtest(prices, params) -> BacktestResult`
(`BacktestResult` contient : le journal des opérations, la série quotidienne de valeur du portefeuille, les statistiques, les signaux non exécutés).

---

## 7. Fonction 4 — Restitution des résultats

### 7.1 Tableau des opérations (journal)

Une ligne par opération, triable et filtrable, exportable en **CSV** (séparateur `;`, décimales à la virgule, pour ouverture directe dans Excel en français) :

| Colonne | Contenu |
|---|---|
| Date | Date d'exécution |
| Action | Nom de la société + ticker |
| Opération | Achat / Vente |
| Motif | Momentum, Breakout, Momentum + Breakout, Stop loss, Stop suiveur, Fin de période |
| Quantité | Nombre d'actions |
| Cours d'exécution | Prix unitaire |
| Montant brut | Quantité × cours |
| Frais | Total des frais de l'opération selon la plateforme choisie (détail courtage / TTF en infobulle) |
| Montant net | Débité ou crédité |
| Plus/moins-value réalisée | Pour les ventes : € et % |
| Liquidités après opération | |
| **Valeur globale du portefeuille** | Après l'opération |

### 7.2 Graphiques (période observée)

1. **Évolution de la valeur globale du portefeuille** (courbe quotidienne), comparée à l'indice de référence ramené au même capital initial. Repères verts (achats) et rouges (ventes) sur la courbe.
2. **Graphique par action** (sélection dans une liste) : cours avec chandeliers ou ligne, points d'achat/vente, et niveaux successifs du stop loss / stop suiveur pendant la détention.
3. **Perte maximale depuis un sommet (drawdown)** : graphique en aires sous la courbe principale.
4. **Répartition** : liquidités vs investi au fil du temps (aires empilées).

Les graphiques sont interactifs (zoom, survol affichant le détail de l'opération).

### 7.3 Indicateurs de synthèse (en haut de l'écran résultats)

- Valeur finale et performance totale (€ et %)
- Performance de l'indice de référence sur la même période
- Nombre d'opérations, nombre de trades gagnants / perdants, taux de réussite
- Gain moyen / perte moyenne par trade
- Plateforme utilisée et **total des frais payés** (en € et en % du capital de départ — important pour un petit capital)
- Perte maximale depuis un sommet (max drawdown)
- Durée moyenne de détention

### 7.4 Sauvegarde et comparaison

- Sauvegarder une simulation (paramètres + résultats) sous un nom.
- Rouvrir une simulation sauvegardée.
- Comparer 2 à 4 simulations : courbes de valeur superposées + tableau des indicateurs côte à côte.

---

## 8. Tests

Écrire des tests `pytest` sur des **séries de prix synthétiques** dont le résultat est connu à la main :
- un breakout déclenche bien l'achat à l'ouverture du lendemain ;
- le stop loss vend au bon prix, y compris en cas de trou de cotation à l'ouverture ;
- le stop suiveur monte avec le cours et ne redescend jamais ;
- les frais (minimum, pourcentage, TTF) sont corrects ;
- le portefeuille ne passe jamais en liquidités négatives ;
- aucune donnée postérieure au jour J n'est utilisée pour une décision prise le jour J.

Le moteur doit être testé **avant** de construire l'interface.

---

## 9. Volet Réel — à préparer, pas à implémenter

Prévoir uniquement :
- une interface `Broker` (`get_positions()`, `get_cash()`, `place_order()`, `cancel_order()`) ;
- l'implémentation `PaperBroker` utilisée par le simulateur.

Le choix du courtier, l'authentification, le mode « signal du soir → ordre à valider manuellement », les garde-fous (plafond par ordre, confirmation obligatoire, journal d'audit) seront spécifiés plus tard dans une section dédiée de ce fichier.

---

## 10. Ordre de développement

1. Squelette du projet (nom, logo, thème, favicon), `pyproject.toml`, `DataProvider` + registre des sources + yfinance + cache Parquet.
2. Calcul des indicateurs (momentum, breakout) + tests.
3. Moteur de simulation (portefeuille, frais, stops, boucle) + tests.
4. Interface Streamlit : onglet Paramétrage (choix de la source, Yahoo seul actif) puis écran Données.
5. Écran Simulation (paramètres) + écran Résultats (tableau, indicateurs, graphiques).
6. Export CSV, sauvegarde SQLite, comparaison de simulations.

À chaque étape : faire tourner les tests, puis montrer comment lancer l'application.

---

## 11. Hors périmètre V1 (pistes pour plus tard)

- Optimisation automatique des paramètres (balayage de plusieurs combinaisons) — avec avertissement sur le sur-ajustement.
- Découpage période d'apprentissage / période de test.
- Données intraday.
- Implémentation des sources EODHD et Interactive Brokers (déjà visibles, grisées, dans l'onglet Paramétrage).
- Prise de bénéfices (take profit) en complément des stops.
- Volet Réel (§9).
