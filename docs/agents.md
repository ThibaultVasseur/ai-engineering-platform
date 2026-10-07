# Agents

Neuf agents aux contrats étroits et typés. Les missions et les listes d'outils autorisés sont
déclarées comme des données dans `app/agents/registry.py` (et servies par `GET /api/v1/agents`) ;
les prompts sont versionnés dans `app/llm/prompts.py` ; les modèles d'entrée et de sortie sont
dans `app/schemas`.

| Agent | Rôle | Entrée → Sortie | Outils | Utilise le LLM |
|---|---|---|---|---|
| Prompt Optimizer | contrôle | demande → `PromptSpec` | — | oui, sauf si la demande est bloquée |
| Planner | contrôle | `PromptSpec` → `Plan` | — | oui |
| Supervisor | contrôle | état du graphe → `SupervisorDecision` | — | non par défaut (`rules`) ; optionnel avec `llm` |
| Research | exécution | `WorkerInput` → `ResearchReport` | `knowledge_search`, `task_lookup`, `calculator` | oui (boucle d'outils) |
| Data | exécution | `WorkerInput` → `DataAnalysis` | `database_read`, `calculator`, `database_write` (sur autorisation explicite) | oui (boucle d'outils) |
| Coding | exécution | `WorkerInput` → `CodeProposal` | `knowledge_search` | oui (boucle d'outils) |
| RAG | exécution | `WorkerInput` → `RagAnswer` | aucun (pipeline de recherche fixe) | oui, sauf si rien de pertinent n'est trouvé |
| Synthesizer | contrôle | `SynthesisInput` → `FinalDraft` | — | oui |
| Critic | relecture | `ReviewInput` → `Review` | — | oui |

## Ce que partagent tous les agents

* **Point d'entrée unique** — `BaseAgent.run` vérifie le budget du run (`MAX_AGENT_CALLS`, durée,
  coût) *avant* toute action, compte l'appel, mesure les tokens et la latence, et émet
  `agent_started` / `agent_completed` / `agent_failed` avec le modèle, la version du prompt et
  la sortie. Chaque appel devient une ligne `agent_runs`.
* **Modèle** — tous les agents LLM utilisent le fournisseur et le modèle configurés
  (`LLM_PROVIDER`, `LLM_MODEL` ; `claude-opus-5-5` par défaut pour Anthropic,
  `offline-simulator-v1` hors ligne) avec `LLM_EFFORT` et `LLM_MAX_OUTPUT_TOKENS`. Chaque appel
  passe par le client budgété, qui applique les limites de coût et de durée du run. Prompts
  révisés après les exécutions réelles : optimiseur `v5`, planificateur `v6`, RAG et
  synthétiseur `v3`, critique `v2` ; les prompts data, research et coding sont en `v1`. Modifier
  un prompt implique d'incrémenter sa version, enregistrée à chaque appel.
* **Sortie structurée** — la réponse doit être valide au regard du modèle Pydantic de l'agent et
  de ses règles métier ; les erreurs de validation sont renvoyées pour réparation, avec au plus
  `LLM_STRUCTURED_MAX_ATTEMPTS` tentatives au total. Un refus n'est jamais « réparé ».
* **Données non fiables** — la partie variable de chaque prompt (demande, documents, résultats
  d'outils, sorties précédentes) est du JSON placé dans `<context>`, avec `</` échappé, sous une
  règle système qui précise qu'il s'agit de données, pas d'instructions.
* **Boucle d'outils** (agents d'exécution dotés d'outils) — au plus `MAX_TOOL_CALLS_PER_AGENT`
  appels par invocation ; les suivants sont refusés avec un message explicite ; la boucle a un
  nombre de tours maximal. Chaque appel passe par les contrôles du registre décrits dans
  [Outils](#outils).
* **Erreurs** — les erreurs du fournisseur, les sorties invalides après le budget de
  réparation, les refus et les erreurs d'agent sont *récupérables* : le nœud enregistre une
  `RunError` et c'est le superviseur ou les routeurs qui décident de la suite (voir
  [architecture.md](architecture.md#gestion-des-erreurs)).

---

## Prompt Optimizer

* **Mission** — transformer une demande brute en spécification structurée et vérifiable, et
  arrêter les demandes dangereuses avant qu'un autre agent ou un outil ne les voie.
* **Entrée** — la demande brute (validée par l'API : 10 à 8 000 caractères après suppression
  des caractères invisibles et de contrôle).
* **Sortie** — `PromptSpec` : `objective`, `context`, `constraints`, `requirements`,
  `deliverables`, `risks`, `acceptance_criteria`, `assumptions`, `clarifying_questions`,
  `needs_knowledge_base`, `feasibility` (`actionable` | `needs_clarification` | `rejected`),
  `rejection_reason`. Une spécification exploitable doit avoir des exigences, des livrables et
  des critères d'acceptation ; une spécification rejetée doit avoir une raison.
* **Outils / permissions** — aucun.
* **Garde-fou** — une détection déterministe d'injection s'exécute d'abord. Risque élevé → une
  spécification `rejected` est renvoyée **sans aucun appel au LLM** (`guardrail_triggered`,
  action `blocked`). Risque plus faible → le résultat de la détection est transmis au modèle
  comme un signal et tracé (action `flagged`).
* **Règles de rédaction** (prompt `v5`, après les exécutions réelles) — conserver l'intention de
  la demande (une question sur « ce qui est » est une recherche dans les documents, jamais une
  tâche de conception), écrire chaque champ dans la langue de l'utilisateur, et formuler des
  critères d'acceptation qu'une réponse honnête peut satisfaire.
* **Limites** — la taille des listes est bornée par le schéma (par exemple au plus 8 critères
  d'acceptation et 5 questions de clarification).
* **Erreurs possibles** — erreur du fournisseur ou spécification invalide après réparations →
  le run s'arrête avant la planification (`abort_reason`, statut final `failed`).
* **Ambiguïté** — une demande floue n'est pas refusée : `needs_clarification` avec des
  hypothèses et des questions explicites ; le résultat final porte un avertissement.

## Planner

* **Mission** — décomposer la spécification en un petit DAG d'étapes, chacune confiée à un agent
  d'exécution.
* **Entrée** — `PromptSpec` et `MAX_PLAN_STEPS`.
* **Sortie** — `Plan` : `rationale` et `steps`, chacune avec `id` (snake_case), `type`
  (`research` | `data` | `coding` | `knowledge`), `title`, `description`, `recommended_agent`,
  `dependencies`, `priority` (1–5), `success_criteria`.
* **Validation** — identifiants uniques, dépendances connues, pas d'autodépendance, pas de cycle
  (algorithme de Kahn, qui fournit aussi l'ordre d'exécution par priorité), au plus
  `MAX_PLAN_STEPS` étapes (règle contextuelle renvoyée pour réparation en cas de violation). Les
  identifiants d'étapes, et les dépendances qui y font référence, sont d'abord normalisés en
  snake_case court (`Design-API` → `design_api`, 40 caractères au plus) : un modèle réel écrivait
  des identifiants trop longs, et l'orthographe d'un identifiant n'a aucun sens métier. Quand la
  spécification indique `needs_knowledge_base: true`, le plan doit contenir une étape
  `knowledge` ou `research` — vérifié dans le code : un plan qui n'en a pas est renvoyé pour
  réparation (un modèle réel avait répondu à une question sur la proposition d'un prestataire
  avec une seule étape de code).
* **Outils / permissions** — aucun.
* **Erreurs possibles** — plan invalide après réparations ou erreur du fournisseur → le run
  s'arrête avant l'exécution.

## Supervisor

* **Mission** — ordonnancer les étapes prêtes, choisir l'agent d'exécution et lui donner ses
  consignes, appliquer les limites, transformer une relecture négative en reprise.
* **Entrée** — l'état du graphe.
* **Sortie** — `SupervisorDecision` : `action` (`dispatch` | `synthesize` | `finalize`),
  `step_id`, `agent`, `reason`, `instructions`.
* **Politique, dans l'ordre** — (1) limite d'étapes du graphe ou dépassement de budget →
  `finalize` avec la limite comme raison ; (2) reprise en attente → les étapes signalées et
  toutes celles qui en dépendent passent en `rework` avec les retours du critique, ou nouvelle
  `synthesize` si seule la synthèse était en cause (compte comme une reprise) ; (3) une étape a
  échoué `MAX_STEP_ATTEMPTS` fois → `finalize` ; (4) prochaine étape prête (dépendances
  terminées), par ordre topologique et priorité → `dispatch` vers l'agent recommandé s'il peut
  exécuter ce type d'étape, sinon vers l'agent par défaut capable de le faire ; (5) toutes les
  étapes terminées → `synthesize` ; (6) plus rien d'exécutable → `finalize`.
* **Capacités** — étapes `research` : research ou RAG ; `knowledge` : RAG ou research ;
  `data` : data ; `coding` : coding.
* **Modèle** — aucun LLM avec la stratégie par défaut `rules`. Avec `SUPERVISOR_STRATEGY=llm`
  (ou `options.supervisor_strategy` pour un run) et plusieurs étapes prêtes, le modèle renvoie un
  `SupervisorChoice` (étape, agent, raison, consignes) validé par la politique ; un choix
  invalide ou un modèle indisponible fait revenir aux règles, et c'est tracé
  (`llm_proposal_rejected`).
* **Outils / permissions** — aucun.
* **Erreurs possibles** — aucune exception : toute situation anormale devient une décision
  `finalize` explicite, avec une raison.

## Agent Research

* **Mission** — rassembler des éléments de preuve avec des outils et rendre des constats
  sourcés.
* **Entrée** — `WorkerInput` : objectif, son étape, contraintes, critères d'acceptation,
  résultats des étapes dont il dépend, retours du critique en cas de reprise, consignes
  optionnelles du superviseur.
* **Sortie** — `ResearchReport` : `summary`, `confidence`, `findings` (affirmation, preuve,
  identifiants de sources), `gaps`.
* **Outils / permissions** — `knowledge_search`, `task_lookup`, `calculator` (tous en lecture).
* **Validation** — les identifiants de sources cités doivent avoir été récupérés pendant ce run.
* **Erreurs possibles** — les erreurs et refus d'outils sont renvoyés au modèle ; une sortie qui
  cite des sources inconnues est renvoyée pour réparation ; un échec persistant devient un échec
  d'étape, relancé jusqu'à `MAX_STEP_ATTEMPTS` fois.

## Agent Data

* **Mission** — analyser les données de la plateforme (tâches, runs, documents, appels
  d'agents) et calculer des métriques.
* **Entrée** — `WorkerInput`.
* **Sortie** — `DataAnalysis` : `summary`, `confidence`, `metrics` (nom, valeur, unité,
  **méthode**), `insights`, `limitations`.
* **Outils / permissions** — `database_read` (requêtes fixes et paramétrées : `count_by_status`,
  `recent`, `summary`), `calculator`. `database_write` figure dans sa liste autorisée mais c'est
  un outil d'ÉCRITURE : il est masqué au modèle et refusé sauf si le run a été lancé avec
  `options.allow_tool_writes=true`, et même dans ce cas il ne peut qu'ajouter une courte note à
  une tâche du même tenant (20 notes au plus par tâche).
* **Validation** — chaque métrique doit indiquer comment elle a été obtenue (requête ou calcul).
* **Erreurs possibles** — comme pour l'agent Research ; `database_write` sans autorisation
  explicite est refusé et enregistré comme `tool_denied`.

## Agent Coding

* **Mission** — concevoir une solution technique : composants, endpoints d'API, modèle de
  données, squelettes de code courts, tests et risques.
* **Entrée** — `WorkerInput`.
* **Sortie** — `CodeProposal` : `summary`, `confidence`, `components`, `api_endpoints` (méthode,
  chemin, rôle), `data_model` (entités et champs), `files` (chemin, langage, rôle, contenu
  ≤ 4 000 caractères), `tests`, `risks`, `source_ids`.
* **Outils / permissions** — uniquement `knowledge_search`. **Ni shell, ni système de fichiers,
  ni exécution de code** : les fichiers proposés sont des données dans la sortie, jamais
  écrits ; leurs chemins doivent être relatifs et sans segment `..`.
* **Validation** — les identifiants de sources cités doivent avoir été récupérés pendant ce run.
* **Erreurs possibles** — comme pour l'agent Research.

## Agent RAG

* **Mission** — répondre à une question précise à partir des seuls documents internes, avec des
  citations vérifiées, et le dire quand les documents ne contiennent pas la réponse.
* **Entrée** — `WorkerInput` (la question est la description de l'étape).
* **Sortie** — `RagAnswer` : `summary`, `confidence`, `answer`, `citations` (identifiant de
  source + citation exacte), `missing_information`.
* **Pipeline** — recherche hybride dans la base de connaissances du tenant (top `RAG_TOP_K`),
  chunks signalés pour injection exclus (`guardrail_triggered`, action `excluded`) ; les sources
  reçoivent une étiquette pour le run (`S1`…) et sont tracées (événement `retrieval`). Si rien de
  pertinent n'est trouvé, l'agent répond « absent des documents » **sans appeler le LLM**.
* **Validation** — chaque citation doit renvoyer à une source récupérée et sa citation doit
  être exacte dans cette source : les mêmes mots dans le même ordre (casse, ponctuation,
  guillemets, tirets et espaces ignorés ; coupures marquées « … » acceptées si chaque fragment
  apparaît dans l'ordre). Une paraphrase ou une traduction est rejetée. Une citation de plus de
  300 caractères est coupée à une frontière de mot et marquée « … » avant vérification (un
  modèle réel citait des passages entiers) ; une quasi-citation est remplacée par la phrase de
  la source qu'elle reproduit (≥ 80 % de ses mots, nombres identiques), si bien que les
  citations affichées sont toujours le texte même de la source. Une réponse doit avoir au moins
  une citation ou une liste explicite des informations manquantes.
* **Outils / permissions** — aucun : la recherche est une étape fixe, pas une décision du
  modèle.
* **Erreurs possibles** — pas de moteur de recherche configuré → erreur d'agent ; citations
  invalides après réparations → échec d'étape.

## Synthesizer

* **Mission** — fusionner les résultats des étapes en une réponse cohérente qui couvre chaque
  critère d'acceptation, dans la langue de la demande d'origine.
* **Entrée** — `SynthesisInput` : demande d'origine (pour sa langue), objectif, livrables,
  critères d'acceptation, hypothèses, questions de clarification, résumés d'étapes (avec les
  sorties complètes), extraits de sources et retours du critique en cas de nouvelle synthèse.
* **Sortie** — `FinalDraft` : `title`, `executive_summary`, `sections` (Markdown),
  `recommendations`, `open_questions`, `criteria_coverage` (critère, couvert ou non, où).
* **Validation** — les citations `[S#]` doivent renvoyer à des sources récupérées pendant ce
  run ; le prompt interdit de présenter comme citation un texte qui ne figure pas mot pour mot
  dans une source ou un résultat d'étape.
* **Outils / permissions** — aucun.
* **Erreurs possibles** — brouillon invalide après réparations ou erreur du fournisseur → le
  run s'arrête (`failed`).

## Critic

* **Mission** — vérifier le brouillon au regard des critères d'acceptation, signaler erreurs et
  risques, et demander des reprises ciblées.
* **Entrée** — `ReviewInput` : objectif, critères d'acceptation, résumés des étapes du plan, le
  brouillon, les étiquettes de sources disponibles, le seuil de réussite et le numéro du tour de
  relecture.
* **Sortie** — `Review` : `status` (`PASS` | `FAIL`), `score` (0–100), `issues` (gravité, étape,
  critère), `suggestions`, `criteria_results`, `rework_steps`.
* **Règles de jugement** (prompt `v2`, après les exécutions réelles) — une réponse qui dit
  clairement que les documents ne contiennent pas une information demandée, sans l'inventer,
  satisfait les critères qui la demandent ; une réponse qui ne traite pas la question posée
  échoue, quelle que soit sa qualité.
* **Politique** (`apply_review_policy`, déterministe, après le modèle) — une citation de source
  jamais récupérée ajoute un problème **critique** ; les cibles de reprise sont restreintes aux
  étapes existantes et complétées à partir des problèmes imputables à une étape ; `PASS`
  seulement si le modèle a dit PASS **et** que le score est ≥ `CRITIC_PASS_THRESHOLD` **et**
  qu'il n'y a aucun problème critique. La politique peut durcir le verdict, jamais l'assouplir.
* **Limites** — au plus `MAX_AGENT_RETRIES` cycles de reprise ; ensuite le run se termine en
  `failed` avec le dernier brouillon et la dernière relecture.
* **Outils / permissions** — aucun.
* **Erreurs possibles** — erreur du fournisseur ou relecture invalide après réparations → le run
  s'arrête (`failed`).

---

## Outils

| Outil | Permission | Arguments | Rôle |
|---|---|---|---|
| `knowledge_search` | lecture | `query` (3–300 caractères), `top_k` (1–8), `category` | Recherche hybride dans la base de connaissances du tenant ; renvoie des extraits étiquetés (`S#`) avec leurs signaux d'injection et une note rappelant que ce contenu est une donnée non fiable. |
| `task_lookup` | lecture | `query` ou `task_id`, `limit` (1–10) | Tâches passées du tenant (id, statut, extrait de la demande, date de création). |
| `calculator` | lecture | `expression` | Arithmétique sur une liste blanche d'arbre syntaxique (nombres, `+ - * /`, `%`, `//`, signes unaires, puissances d'exposant ≤ 64, parenthèses) ; pas d'`eval` ; résultats plafonnés à 1e18. Délai maximal 2 s. |
| `database_read` | lecture | `entity` (`tasks` \| `runs` \| `documents` \| `agent_runs`), `query` (`count_by_status` \| `recent` \| `summary`), `limit` (1–50) | Requêtes fixes, paramétrées et limitées au tenant. Jamais de SQL brut. |
| `database_write` | **écriture** | `action` (`annotate_task`), `task_id`, `note` (5–500 caractères) | Ajoute une note à une tâche du même tenant (20 au plus par tâche). Masqué et refusé sans `allow_tool_writes`. |

Chaque appel passe par `BoundToolExecutor.execute`, qui applique dans l'ordre : l'outil existe →
il est dans la liste autorisée de l'agent appelant → un outil d'écriture exige l'autorisation
explicite du run → les arguments sont validés par le modèle Pydantic de l'outil → la fonction
s'exécute avec un délai maximal (`TOOL_TIMEOUT_SECONDS`, 10 s par défaut) → la sortie est
sérialisée et tronquée (`TOOL_MAX_OUTPUT_CHARS`). Chaque appel, autorisé ou refusé, est
enregistré avec son statut, ses arguments masqués, un aperçu de la sortie et sa latence, et
tracé (`tool_call` / `tool_denied`). Une fonction qui échoue de façon inattendue renvoie une
erreur générique : aucune trace de pile ni requête SQL ne parvient au modèle.

## Ajouter un agent

1. Définir ses modèles d'entrée et de sortie dans `app/schemas` (champs stricts et bornés ;
   règles métier sous forme de validateurs).
2. Ajouter un `PromptTemplate` versionné dans `app/llm/prompts.py` incluant
   `UNTRUSTED_DATA_RULE`.
3. Déclarer son `AgentProfile` (mission, rôle, **liste d'outils autorisés**, schéma de sortie)
   dans `app/agents/registry.py`.
4. Hériter de `BaseAgent` et implémenter `execute` avec `_generate` ou `_tool_loop`.
5. Le brancher dans le graphe (nœud + routeur) et dans `build_agent_suite` ; ajouter un handler
   hors ligne pour que les tests et l'évaluation puissent l'exercer sans clé ; ajouter des cas
   d'évaluation.
