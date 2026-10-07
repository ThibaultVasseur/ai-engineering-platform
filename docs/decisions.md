# Décisions d'architecture (ADR)

Chaque décision présente le contexte, la décision, les alternatives envisagées et les
conséquences, coûts compris. Quand un choix était ambigu, l'option la plus simple qui garde le
système honnête et testable a été retenue, et le compromis est consigné ici.

| # | Décision | Statut |
|---|---|---|
| [001](#adr-001--fastapi-pour-lapi-http) | FastAPI pour l'API HTTP | acceptée |
| [002](#adr-002--postgresql-comme-unique-système-de-référence) | PostgreSQL comme unique système de référence | acceptée |
| [003](#adr-003--pgvector-pour-la-recherche-vectorielle) | pgvector pour la recherche vectorielle | acceptée |
| [004](#adr-004--rag-hybride-avec-citations-vérifiées) | RAG hybride avec citations vérifiées | acceptée |
| [005](#adr-005--langgraph-pour-lorchestration) | LangGraph pour l'orchestration | acceptée |
| [006](#adr-006--multi-agents-avec-des-agents-au-moindre-privilège) | Multi-agents avec des agents au moindre privilège | acceptée |
| [007](#adr-007--des-sorties-structurées-partout) | Des sorties structurées partout | acceptée |
| [008](#adr-008--le-critique-comme-contrôle-qualité-avec-reprises-bornées) | Le critique comme contrôle qualité, avec reprises bornées | acceptée |
| [009](#adr-009--docker-compose-pour-lenvironnement-local) | Docker Compose pour l'environnement local | acceptée |
| [010](#adr-010--langfuse-comme-observabilité-optionnelle) | Langfuse comme observabilité optionnelle | acceptée |
| [011](#adr-011--une-couche-llm-légère-et-indépendante-du-fournisseur) | Une couche LLM légère et indépendante du fournisseur | acceptée |
| [012](#adr-012--superviseur-déterministe-conseil-llm-optionnel) | Superviseur déterministe, conseil LLM optionnel | acceptée |
| [013](#adr-013--fournisseur-hors-ligne-déterministe) | Fournisseur hors ligne déterministe | acceptée |
| [014](#adr-014--la-sécurité-au-niveau-des-lignes-comme-défense-en-profondeur) | La sécurité au niveau des lignes comme défense en profondeur | acceptée |
| [015](#adr-015--embeddings-par-hachage-par-défaut-512-dimensions) | Embeddings par hachage par défaut, 512 dimensions | acceptée |
| [016](#adr-016--runs-en-arrière-plan-dans-le-processus) | Runs en arrière-plan dans le processus | acceptée (à revoir) |
| [017](#adr-017--uv-pour-la-gestion-des-dépendances) | uv pour la gestion des dépendances | acceptée |

---

## ADR 001 — FastAPI pour l'API HTTP

**Contexte.** La plateforme expose des tâches, des runs, leurs événements, l'ingestion de
connaissances, un catalogue d'agents et des évaluations. L'API doit valider chaque contenu
reçu, se documenter elle-même et être asynchrone de bout en bout (base de données, fournisseurs
LLM et graphe d'orchestration sont tous limités par les entrées-sorties).

**Décision.** FastAPI avec une fabrique d'application (`app.main:create_app`). Au démarrage
(lifespan), un conteneur de dépendances unique est construit (la racine de composition,
`app/container.py`) ; les routes sont fines et délèguent aux services ; l'authentification, la
résolution du tenant et la limitation de débit sont des dépendances FastAPI ; les identifiants
de requête, les en-têtes de sécurité et la taille maximale des corps sont des middlewares ASGI
purs.

**Alternatives.** Django REST Framework (plus lourd, ORM d'abord synchrone), Litestar
(écosystème plus réduit), Flask + extensions (pas de validation asynchrone native).

**Conséquences.** Les modèles Pydantic sont partagés par l'API, les agents et la couche LLM, et
le document OpenAPI (`/docs`) est toujours synchronisé avec le code. La fabrique et le conteneur
injectable permettent à chaque test de construire une application isolée avec sa propre
configuration.

## ADR 002 — PostgreSQL comme unique système de référence

**Contexte.** La plateforme stocke des tâches, des runs, un journal d'événements en ajout seul,
les exécutions de chaque agent, des documents, des chunks avec leurs embeddings et leurs
vecteurs plein texte, et des rapports d'évaluation. Tout est limité au tenant.

**Décision.** PostgreSQL 17 pour tout, via SQLAlchemy 2 (asynchrone, asyncpg) et des migrations
Alembic. JSONB pour les contenus semi-structurés (plans, résultats finaux, contenus
d'événements, métadonnées des chunks), avec des contraintes de contrôle sur les colonnes de
statut et des index conçus pour les chemins d'accès (`tenant_id` en premier).

**Alternatives.** Une base vectorielle séparée (Qdrant, Pinecone, Weaviate) à côté de
PostgreSQL ; une base orientée documents pour les runs et les événements.

**Conséquences.** Un seul stockage transactionnel : un document et ses chunks sont écrits dans
la même transaction, la sécurité au niveau des lignes couvre les vecteurs comme les lignes
métier, les sauvegardes et l'exploitation restent simples. La contrepartie : la recherche
vectorielle évolue avec PostgreSQL (voir ADR 003).

## ADR 003 — pgvector pour la recherche vectorielle

**Contexte.** Le RAG a besoin d'une recherche approchée des plus proches voisins, filtrée par
tenant et par métadonnées.

**Décision.** Colonne `vector(512)` avec un index HNSW (`m=16, ef_construction=64`, distance
cosinus). Les requêtes activent `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) pour
qu'une recherche filtrée continue jusqu'à trouver assez de lignes au lieu de renvoyer
silencieusement moins de résultats.

**Alternatives.** IVFFlat (a besoin de données d'apprentissage et d'une reconstruction quand le
corpus grossit) ; une base vectorielle dédiée (un service de plus, un autre modèle de
permissions, des données dupliquées).

**Conséquences.** Les vecteurs vivent à côté de leurs documents, sous les mêmes politiques RLS.
HNSW coûte de la mémoire et ralentit les insertions ; c'est acceptable pour une base de
connaissances bien plus souvent lue qu'écrite. Changer de dimension exige une migration (voir
ADR 015).

## ADR 004 — RAG hybride avec citations vérifiées

**Contexte.** Les réponses doivent s'appuyer sur les documents internes, les citer et le dire
quand les documents ne contiennent pas la réponse. Les documents sont aussi une surface
d'attaque (injection de prompt indirecte).

**Décision.** De la génération augmentée par recherche plutôt que du fine-tuning ou « tous les
documents dans le contexte » : découpage tenant compte de la structure, avec recouvrement ;
embeddings de `titre + chunk` ; **recherche hybride** (recherche vectorielle ∥ recherche plein
texte PostgreSQL, fusionnées par Reciprocal Rank Fusion) ; exclusion des chunks signalés comme
injection à l'ingestion ; reclassement lexical ; étiquettes de sources stables par run
(`[S1]`…). L'agent RAG vérifie que chaque citation renvoie à une source récupérée et qu'elle y
figure mot pour mot — les citations affichées sont, par construction, le texte même de la
source : une quasi-citation est remplacée par la phrase qu'elle reproduit — et il répond sans
appeler le LLM quand rien de pertinent n'est trouvé. Détails dans [rag.md](rag.md).

**Alternatives.** Recherche uniquement vectorielle (rate les codes et identifiants exacts comme
`DEV-AAAA-NNNNN`) ; fine-tuning (connaissances figées, pas de citations) ; long contexte rempli
de documents (coût, latence, aucune isolation du contenu non fiable).

**Conséquences.** La qualité de la recherche est mesurable (critère d'évaluation
`expected_sources`) et explicable (chaque résultat porte son score vectoriel, son rang plein
texte et son score fusionné). Le reclassement est volontairement simple et remplaçable derrière
le protocole `Reranker`.

## ADR 005 — LangGraph pour l'orchestration

**Contexte.** Le workflow a des étapes explicites (optimiser → planifier → superviser →
exécuter → synthétiser → relire → reprendre ou terminer), un routage conditionnel, des boucles
qui doivent être bornées et un état qui doit pouvoir être inspecté après coup.

**Décision.** Un `StateGraph` LangGraph avec un `GraphState` typé (TypedDict de valeurs
Pydantic) et des réducteurs pour les historiques (`operator.add`) et l'état par étape
(`merge_steps`). Le graphe est compilé **une seule fois** ; les dépendances propres à un run
(tenant, budget, traceur, exécuteur d'outils, agents) sont injectées via le contexte d'exécution
de LangGraph (`context_schema=RunContext`). Les routeurs sont des fonctions pures de l'état. Le
diagramme Mermaid est généré à partir du graphe compilé (`python -m app.cli --graph`) : la
documentation ne peut pas diverger du code.

**Alternatives.** Une boucle écrite à la main (simple, mais il faudrait reconstruire le routage,
le streaming et la fusion d'état) ; l'`AgentExecutor` de LangChain ou des frameworks
conversationnels à la CrewAI / AutoGen (flux de contrôle caché dans les prompts, plus difficile
à borner et à tester).

**Conséquences.** Chaque transition est visible et testable unitairement ; `astream` fournit
les mises à jour des nœuds pour suivre la progression ; `recursion_limit` sert de garde-fou de
dernier recours. Seul le moteur de graphe de LangGraph est utilisé — pas les wrappers de modèles
de LangChain (voir ADR 011).

## ADR 006 — Multi-agents avec des agents au moindre privilège

**Contexte.** Un agent généraliste doté de tous les outils accumulerait le contexte, les
permissions et les modes de défaillance. Le cahier des charges demande un superviseur et des
agents spécialisés.

**Décision.** Neuf rôles aux contrats étroits : Prompt Optimizer, Planner, Supervisor, quatre
agents d'exécution (Research, Data, Coding, RAG), Synthesizer et Critic. Chaque agent a une
entrée typée, une sortie typée et une liste explicite d'outils autorisés, déclarée comme une
donnée (`app/agents/registry.py`) et appliquée par le registre d'outils. Les agents d'exécution
ne reçoivent que leur étape, les critères d'acceptation et les résultats des étapes dont ils
dépendent.

**Alternatives.** Un seul agent ReAct doté de tous les outils ; des conversations libres entre
agents.

**Conséquences.** Des prompts plus courts, une évaluation et un traçage par agent, et un rayon
d'impact limité par les permissions : l'agent Coding ne peut toucher ni un fichier ni un shell,
et seul l'agent Data peut même voir l'outil d'écriture. La contrepartie : davantage d'appels LLM
par run (7 à 12 par run terminé dans l'évaluation hors ligne, 5 à 19 avec `gpt-4.1-mini`),
que les limites de budget maintiennent bornés.

## ADR 007 — Des sorties structurées partout

**Contexte.** Les agents transmettent leurs résultats à du code (routeurs, superviseur,
politique du critique, API), pas à des humains. Il faudrait sinon analyser du texte libre avec
des heuristiques.

**Décision.** Chaque sortie LLM est un modèle Pydantic (`PromptSpec`, `Plan`, `ResearchReport`,
`DataAnalysis`, `CodeProposal`, `RagAnswer`, `FinalDraft`, `Review`, `SupervisorChoice`). Un
schéma JSON strict est dérivé de chaque modèle pour le décodage contraint
(`additionalProperties: false`, toutes les propriétés requises, mots-clés non pris en charge
retirés). Les contraintes retirées (longueurs, nombres d'éléments, bornes, motifs) sont
réexpliquées en clair dans la description de chaque champ, pour que le modèle les connaisse
avant de répondre — une leçon de la première exécution réelle, où un modèle qui ignorait une
limite de 300 caractères citait des paragraphes entiers. La réponse est ensuite validée par le
modèle Pydantic *complet*, y compris les règles métier que le schéma ne peut pas exprimer (un
plan est un DAG, les citations renvoient à des sources récupérées et sont exactes, les
métriques indiquent leur méthode). Les erreurs de validation sont renvoyées au modèle pour un
nombre borné de réparations (`LLM_STRUCTURED_MAX_ATTEMPTS`) ; un refus n'est jamais
« réparé ».

Les variations sans signification sont normalisées avant validation au lieu de faire échouer un
run : identifiants d'étapes (`Design-API` → `design_api`), étiquettes de sources écrites `[S1]`,
méthode ou paramètres de requête dans le chemin d'un endpoint, `./` dans un chemin de fichier,
citation exacte trop longue (coupée à une frontière de mot, marquée « … », puis vérifiée). Tout
ce qui a du sens — une citation, une référence à une source, une dépendance, un score — est
validé strictement.

**Alternatives.** Mode JSON sans schéma ; analyse de texte libre par expressions régulières ;
appels de fonctions uniquement.

**Conséquences.** Les sorties invalides échouent bruyamment et sont tracées
(`structured_output_repaired`) au lieu de se propager. Les champs `dict` libres sont refusés
par le générateur de schémas, ce qui garde chaque contrat explicite.

## ADR 008 — Le critique comme contrôle qualité, avec reprises bornées

**Contexte.** Un pipeline qui renvoie toujours « succès » n'est pas digne de confiance. La
réponse doit être vérifiée au regard des critères d'acceptation produits au début du run.

**Décision.** Un agent Critic relit le brouillon synthétisé (statut `PASS|FAIL`, score 0–100,
problèmes avec gravité et étape éventuelle, suggestions, résultat par critère, étapes à
reprendre). Une politique déterministe a ensuite le dernier mot et ne peut que **durcir** le
verdict : une citation de source jamais récupérée devient un problème critique, les cibles de
reprise sont restreintes aux étapes existantes, et `PASS` exige le PASS du modèle, un score
≥ `CRITIC_PASS_THRESHOLD` et aucun problème critique. En cas de FAIL, le superviseur reprend les
étapes signalées **et celles qui en dépendent** (ou relance seulement la synthèse si aucune
étape n'est en cause), au plus `MAX_AGENT_RETRIES` fois. Un run dont la réponse n'est jamais
acceptée se termine en `FAILED`, avec le dernier brouillon et la dernière relecture joints.

**Alternatives.** Pas de relecture ; auto-réflexion à l'intérieur de chaque agent ; « recommencer
jusqu'à ce que ce soit bon » sans limite.

**Conséquences.** L'acceptation est explicite (`accepted: true|false`) et expliquée. Un critique
LLM n'est pas un oracle : les exécutions réelles l'ont montré (95 pour une réponse à côté de la
question, 30 pour une réponse honnête sur une information absente), d'où des règles de jugement
explicites dans son prompt (v2), et la suite d'évaluation contrôle ses décisions sur des cas
connus.

## ADR 009 — Docker Compose pour l'environnement local

**Contexte.** Un relecteur doit pouvoir démarrer la plateforme en une commande, sans installer
PostgreSQL ni pgvector.

**Décision.** Un Dockerfile multi-étapes (construction avec uv → image d'exécution légère,
utilisateur non root, health check) et un fichier Compose avec `pgvector/pgvector:pg17` et
l'API. Un script d'initialisation crée un rôle `app` non superutilisateur et les bases ; le
point d'entrée de l'API applique les migrations (avec nouvelles tentatives) et peut ingérer le
corpus de démonstration.

**Alternatives.** De simples instructions d'installation locale ; des manifestes Kubernetes
(hors périmètre).

**Conséquences.** `docker compose up --build` donne un environnement fonctionnel avec les
données de démonstration. Le fichier Compose sert au développement et aux démonstrations, pas à
un déploiement en production.

## ADR 010 — Langfuse comme observabilité optionnelle

**Contexte.** Les runs multi-agents ont besoin de traces : quel agent a tourné, avec quelle
version de prompt et quel modèle, combien de tokens, quels outils, combien de temps, ce qu'a
décidé le critique.

**Décision.** L'application émet des `TraceEvent` typés vers une interface de traceur. Les
destinations se composent : PostgreSQL (`run_events`, `agent_runs`) et logs JSON structurés
toujours ; Langfuse quand les clés `LANGFUSE_*` sont configurées et l'extra `observability`
installé. Le traceur Langfuse convertit les événements en observations typées (chaîne, agent,
génération, outil, recherche, garde-fou, évaluateur) et enregistre le score du critique ; son
hook `mask` applique le même masquage des secrets que les logs. Une destination en échec ne
fait jamais échouer un run.

**Alternatives.** Langfuse comme dépendance obligatoire ; OpenTelemetry seul ; LangSmith.

**Conséquences.** La plateforme démarre et reste entièrement traçable sans aucun service
externe ; l'API expose le flux d'événements (`GET /api/v1/runs/{id}/events`). La conversion
vers Langfuse est testée unitairement avec un faux client ; elle n'a pas été exercée contre un
vrai serveur Langfuse.

## ADR 011 — Une couche LLM légère et indépendante du fournisseur

**Contexte.** Le fournisseur doit être interchangeable (Claude, tout endpoint compatible OpenAI,
hors ligne), chaque appel doit être compté sur le budget du run, et les fonctionnalités propres
à chaque fournisseur comptent pour la qualité et le coût.

**Décision.** Une petite couche interne (`app/llm`) : des types neutres (`Message`, `ToolCall`,
`LLMRequest`, `LLMResponse`, `Usage`, `StopReason`), un adaptateur par fournisseur, et un
wrapper de comptage qui vérifie le budget du run avant chaque appel et émet des événements
`llm_call` avec tokens, latence, coût et version de prompt. L'adaptateur Claude (SDK officiel
`anthropic`) utilise par défaut `claude-opus-5-5`, fixe explicitement `output_config.effort`,
utilise `output_config.format` (schéma JSON) et des outils stricts, laisse `tool_choice` sur
`auto`, réémet tels quels les blocs de contenu de l'assistant dans les boucles d'outils
(historique en ajout seul), active la mise en cache automatique des prompts et, quand
`LLM_REFUSAL_FALLBACK=true` (défaut), laisse l'API relancer côté serveur un refus de sécurité
sur le modèle de repli recommandé. L'adaptateur compatible OpenAI utilise Chat Completions avec
des fonctions strictes et `response_format: json_schema`.

**Alternatives.** Les wrappers de modèles de chat de LangChain (une abstraction de plus entre le
code et des fonctionnalités comme l'effort, le repli sur refus ou la réémission de contenu) ;
LiteLLM.

**Conséquences.** Maîtrise complète des requêtes, des erreurs et du comptage, au prix de la
maintenance de deux adaptateurs. L'application de la limite de coût exige un tarif pour le
modèle configuré : les modèles Claude ont des tarifs publics intégrés ; tout autre modèle exige
`LLM_PRICE_INPUT_PER_MTOK` / `LLM_PRICE_OUTPUT_PER_MTOK`, et la plateforme refuse de démarrer
sans eux plutôt que d'appliquer une limite de coût calculée à 0 $. Le corps des erreurs des
fournisseurs est journalisé côté serveur (masqué) et jamais renvoyé aux clients de l'API. Les
deux adaptateurs sont testés avec les vrais types de réponse des SDK et de faux transports ;
l'adaptateur compatible OpenAI a en outre tourné en réel avec `gpt-4.1-mini` (CLI et cinq
évaluations complètes), tandis que **l'adaptateur Anthropic n'a pas encore été exercé contre
l'API réelle**.

## ADR 012 — Superviseur déterministe, conseil LLM optionnel

**Contexte.** Ordonnancer un DAG, appliquer des limites et transformer une relecture en reprise
sont des problèmes déterministes. Les confier à un modèle ajoute du coût et du
non-déterminisme là où la justesse compte le plus.

**Décision.** Le superviseur est une politique : d'abord les limites strictes, puis les
reprises, puis les étapes en échec, puis la prochaine étape prête (dépendances satisfaites),
confiée à un agent d'exécution capable de la traiter. Avec `SUPERVISOR_STRATEGY=llm`, un modèle
peut choisir parmi plusieurs étapes prêtes et donner des consignes à l'agent ; sa proposition
est validée par la même politique et écartée si elle est invalide (« le LLM propose, la
politique dispose »). Quand il n'y a qu'une option, aucun appel LLM n'est fait.

**Alternatives.** Un superviseur entièrement piloté par un LLM (courant dans les démos, difficile
à borner).

**Conséquences.** Le flux de contrôle est reproductible et testable ; les propositions LLM
rejetées sont tracées (`llm_proposal_rejected`). Le jugement du modèle est utilisé là où il
apporte de la valeur : le contenu du plan, le travail et la relecture.

## ADR 013 — Fournisseur hors ligne déterministe

**Contexte.** La CI, les tests et les relecteurs doivent pouvoir exécuter tout le pipeline sans
clé d'API, gratuitement et avec des résultats reproductibles.

**Décision.** Un fournisseur `offline` implémente la même interface `LLMClient` avec des
handlers à base de règles pour chaque prompt (optimiseur, planificateur, agents d'exécution,
synthétiseur, critique). Il lit les mêmes contenus `<context>` qu'un vrai modèle, appelle les
mêmes outils par la même boucle d'outils et renvoie du JSON validé par les mêmes schémas.

**Alternatives.** Des réponses enregistrées (fragiles dès que les prompts changent) ; des
simulacres au niveau des agents (ils court-circuiteraient la couche LLM, la boucle d'outils et
la validation).

**Conséquences.** Le graphe complet — outils, RAG, critique, reprises, limites, garde-fous —
tourne en CI et dans `docker compose` sans secret. Le fournisseur hors ligne ne raisonne pas :
il valide la mécanique, pas la qualité d'un modèle. Les scores d'évaluation obtenus hors ligne
sont un signal de non-régression, pas un benchmark de qualité.

## ADR 014 — La sécurité au niveau des lignes comme défense en profondeur

**Contexte.** Une isolation des tenants assurée uniquement par les requêtes applicatives cède dès
qu'une requête oublie son filtre.

**Décision.** Chaque requête filtre explicitement sur `tenant_id` **et** chaque table a la
sécurité au niveau des lignes activée et *forcée*, avec une politique `tenant_isolation` sur
`current_setting('app.tenant_id', true)`. L'application se connecte avec un rôle non
superutilisateur sans `BYPASSRLS` ; chaque transaction fixe le tenant avec
`set_config(..., true)`. Un paramètre absent donne NULL : aucune ligne n'est visible (fermé par
défaut).

**Alternatives.** Des filtres applicatifs seuls ; un schéma ou une base par tenant.

**Conséquences.** Les tests d'intégration montrent qu'une session du tenant A ne peut ni lire ni
écrire les lignes du tenant B, même avec une requête écrite à la main. Le coût : un
`set_config` par transaction et la discipline de toujours ouvrir les sessions via
`Database.session(tenant_id)`.

## ADR 015 — Embeddings par hachage par défaut, 512 dimensions

**Contexte.** Des embeddings sont nécessaires en CI et dans l'environnement Docker par défaut,
où aucun modèle ne peut être téléchargé ni appelé.

**Décision.** Un `HashingEmbedder` déterministe (hachage de caractéristiques signé des
unigrammes et bigrammes, accents et pluriels ramenés à une forme commune, normalisation L2,
512 dimensions) est utilisé par défaut ; un `OpenAICompatibleEmbedder` (OpenAI, Ollama, vLLM…)
est configurable et demande explicitement 512 dimensions. La colonne est en `vector(512)`.

**Alternatives.** Un modèle sentence-transformers local (gros téléchargement, CI plus lente) ;
uniquement un modèle hébergé (exige une clé).

**Conséquences.** Une recherche reproductible dans les tests. Le hachage ne capte que la
similarité lexicale (pas de synonymes) ; la recherche hybride et le reclassement compensent, et
la production devrait utiliser un vrai modèle d'embeddings. Changer de modèle implique de
revectoriser le corpus ; le nom du modèle est enregistré dans les métadonnées de chaque
document.

## ADR 016 — Runs en arrière-plan dans le processus

**Contexte.** Un run dure de quelques secondes à quelques minutes ; l'API doit répondre
immédiatement (`202`) et laisser les clients interroger le statut et les événements.

**Décision.** Les runs s'exécutent comme des tâches asyncio dans le processus de l'API, au plus
`MAX_CONCURRENT_RUNS` à la fois, avec une progression persistée (statut de la tâche, plan,
événements) au fil du graphe. `wait=true` exécute de façon synchrone pour les démonstrations et
les tests. L'annulation est prise en charge ; à l'arrêt du service, les runs en cours sont
annulés et marqués `CANCELLED`.

**Alternatives.** Une file d'attente durable avec des workers (Celery, Arq, une file adossée à
PostgreSQL) ou un checkpointer LangGraph pour reprendre les runs.

**Conséquences.** Aucune infrastructure supplémentaire, mais un redémarrage interrompt les runs
en cours et la concurrence est propre à chaque instance. C'est la première chose à changer pour
la production (voir le README, « Évolutions possibles »).

## ADR 017 — uv pour la gestion des dépendances

**Contexte.** Les constructions doivent être reproductibles en local, en CI et dans Docker.

**Décision.** `uv` avec un `uv.lock` commité ; la CI et la construction Docker utilisent
`--frozen`. Les outils de développement (pytest, ruff, mypy, pre-commit) sont dans un groupe de
dépendances ; Langfuse est un extra optionnel. Les hooks pre-commit exécutent les versions
verrouillées des outils via `uv run`.

**Alternatives.** pip + fichiers requirements ; Poetry.

**Conséquences.** Un seul fichier de verrouillage pour tous les environnements et des
installations rapides. Les contributeurs doivent installer `uv` (un binaire unique).
