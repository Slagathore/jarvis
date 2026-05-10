<!-- markdownlint-disable -->
# §22+ Generalization Patch — v4 (FINAL)

Supersedes v1, v2, and v3 entirely for all conflicting EDITs. v4 is self-contained — apply this patch alone against the original `new_2.md` and ignore prior versions.

The shape of v4 in one paragraph: §22 grows pet-by-name into a generic animal tracking subsystem with the real Cole/Anna/Jeff household lineup and a richer ownership/affinity model; §13's human state machine grows a `PRESENT_OFF_FRAME` state and trajectory-aware door logic; §29 (new) is a multi-alarm subsystem (Fire, Cat-Escape, Door-Open) with priority-based concurrent audio, condition-clearance as the primary resolve path, and voice-silence + visual-confirmation as secondary paths; §30 (new) is the multi-wake-word persona system; §31 (new) is the unified notifications dispatcher (ntfy + Telegram + Home Assistant). §28 grows tunables for all of it. Schema additions are folded into the relevant sections and listed together in §32.

---

## EDIT 1 — §22 title and opening paragraph

**Anchor:** lines 3044–3046.

**REPLACE** the heading and intro paragraph **WITH:**

```markdown
## 22. Phase 4: Pets by Name (general)

Phase 4 adds named animals. The canonical and default-enabled species are `cat` and `dog`; other animals are added by extending the `tracked_species` whitelist and declaring instances in `pets.<species>` config blocks. Architecturally nothing new at the entity layer — every named animal is a `WorldEntity(entity_type=<species>, person_id=None)` and rides through the same association → state-machine → event-emit pipeline as people. What's species-specific is the *descriptor pipeline* (color/size/coat for cats; size/breed-class/coat for dogs), the per-species cost function (Section 13's `_cat_pair_cost` becomes a `_animal_pair_cost(species, ...)` dispatcher), and the cold-start protocol (each species cold-starts independently against its own resident lineup). The bootstrap flow promotes config-declared animals to entity rows on first run with `is_resident=true`; animals detected for tracked species but with no matching declaration become *transient* entities with `is_resident=false` and an auto-generated handle, and are reaped after `visiting_animal_retention_minutes` of departed status. Animals detected for *non-whitelisted* species (bear, sheep, giraffe — yes, COCO has those) ride the bus as raw observations but never become entities. Outdoor-only animals (Scooter the sulcata tortoise) are out of scope for the world model and are handled by the OutdoorObserver subsystem in §22.10.
```

---

## EDIT 2 — §22.0 Tracked species and visitor distinction (NEW SUBSECTION)

**Anchor:** insert immediately after the opening paragraph, before §22.1.

**INSERT:**

```markdown
### 22.0 Tracked species and the visitor distinction

Two orthogonal switches govern whether a detected animal becomes a tracked entity:

**1. Species whitelist.** `world_model.tracked_species` is a config list. Default for this household: `[cat, dog]`. A YOLO detection whose class is not in the list never reaches the entity layer — it appears in `world.observation` events for debugging but no `WorldEntity` is created or updated.

**2. Resident flag.** `world_entities.is_resident BOOLEAN NOT NULL DEFAULT true`. Two paths set this:

- **Declared in config** (`pets.cats[*]`, `pets.dogs[*]`) → `is_resident=true`. These are the family pets. Permanent entity rows, full behavioral profiles, anomaly scoring, survive across restarts.
- **Auto-discovered at runtime** (whitelisted species, no config match) → `is_resident=false`. Transient handle like `visiting_dog_a3f2`, tracked while present, reaped `visiting_animal_retention_minutes` after entering `DEPARTED` state.

Promotion from transient to resident is a **deliberate config edit + restart**, never automatic. See §22.11 for the workflow.

**Visiting animals are excluded from anomaly scoring and pattern mining.** Same flag check as the human-side flag from §25 — `if not entity.is_resident: skip`. Same column, same logic.

The IdentityManager interface generalizes from `enroll_cat(...)` to:

```python
def enroll_animal(
    self,
    species: str,
    name: str,
    image_crops: list[np.ndarray],
    config_metadata: dict | None = None,
) -> int:
    ...
```

`enroll_animal` dispatches to a per-species `DescriptorExtractor` (cat: color histogram + pattern + size-bin + coat texture; dog: same primitives plus breed-class classifier). Adding a new species = registering one more subclass.
```

---

## EDIT 3 — §22.0a Multi-resident households and affinity (NEW SUBSECTION)

**Anchor:** insert after §22.0.

**INSERT:**

```markdown
### 22.0a Multi-resident households, ownership, and affinity

The household has three human residents (Cole, Anna, Jeff). Pets have one *household owner* (legal/practical owner) and zero or more *affinities* (preferences for specific people in specific contexts). These are distinct concepts that early drafts of this spec collapsed into a single field; v4 separates them.

**Three fields, three roles:**

| Field | Required? | Meaning | Used for |
|---|---|---|---|
| `household_owner` | yes | Whose pet, legally/practically. | Default attribution of pet care notifications, vet records reference, fallback persona phrasing. |
| `unmonitored_home` | no | An unmonitored room this entity defaults to when not observed. | World model `IN_HOUSE_UNMONITORED` reasoning. Set only for pets whose home_room is unmonitored space (Velcro→jeff_room). |
| `affinities` | no | List of person-specific preferences with strength and context. | Persona phrasing, behavioral cost-function priors, sleep-partner detection. |

**Affinity schema:**

```python
@dataclass
class Affinity:
    person: str          # resident id (cole/anna/jeff)
    strength: str        # 'low' | 'medium' | 'high'
    contexts: list[str]  # one or more of: sleeping, physical_contact,
                         # rubbing, proximity_general, authority, feeding
```

**Schema migration:**

```sql
-- migrations/20260512_residents_and_pets_v4.sql

ALTER TABLE world_entities
  ADD COLUMN household_owner_id INTEGER REFERENCES persons(id) ON DELETE SET NULL;

ALTER TABLE world_entities
  ADD COLUMN unmonitored_home_room TEXT NULL;

CREATE TABLE pet_affinities (
  pet_entity_id   INTEGER NOT NULL REFERENCES world_entities(id) ON DELETE CASCADE,
  person_id       INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  strength        TEXT NOT NULL CHECK (strength IN ('low','medium','high')),
  contexts        TEXT NOT NULL,  -- comma-separated tokens
  PRIMARY KEY (pet_entity_id, person_id)
);

CREATE INDEX idx_pet_affinities_pet ON pet_affinities(pet_entity_id);
```

**Why affinity is a list of contexts and not a single number.** Spooky prefers Cole *for sleeping* but is otherwise neutral. A scalar "Cole-affinity = 0.8" can't express that. The contexts model lets the persona answer *"Spooky's on the bed"* with *"yeah, that's expected"* (sleep context, high affinity to Cole) without also predicting that Spooky should follow Cole around all day.

**Persona phrasing.** When the persona refers to a pet, it picks the form that matches the active context. Velcro is rubbing on Anna → *"Velcro is rubbing on Anna"* (active high-affinity-rubbing context with Anna), not *"Jeff's cat Velcro is rubbing on Anna"* (technically true but pedantic). When no active context applies, the persona uses neutral phrasing — *"Velcro is in the kitchen"* — without invoking ownership or affinity at all. Ownership phrasing is reserved for pet-care concerns: *"I haven't seen Jeff's cat Velcro in 8 hours, you might let Jeff know."*

**Anomaly scoring.** `household_owner` provides a small soft prior — a pet in their owner's room at unusual hours is slightly less anomalous than the same pet in a non-owner's room. Strength is small (≤1.0 on the 0–10 scale). Affinity does *not* feed anomaly scoring directly; it's persona-phrasing and behavioral-prediction only.
```

---

## EDIT 4 — §22.1 Disambiguation analysis (REWRITE)

**Anchor:** §22.1, full section.

**REPLACE WITH:**

```markdown
### 22.1 The lineup, and why disambiguation is hard

Six cats (five Cole's, one Jeff's) and two dogs (both Cole's). Plus Scooter, handled in §22.10.

| Name | Color class | Distinguishing features | Size | Home room | Owner |
|---|---|---|---|---|---|
| **Socks** | tuxedo | unique silhouette in lineup | medium | office | Cole |
| **Sneaky** | silver-tabby | blue eyes, long coat, lynx-point, "patient hunter" | medium-large fluff | cyclic (see below) | Cole |
| **Spooky** | black | fluffy/curly coat, calm, sleeps next to Cole | medium | bedroom | Cole |
| **Velcro** | black | straight/sleek coat, fridge-perch, fast/spirited, hates Summer | medium-large | jeff_room (unmonitored) | Jeff |
| **Sparta** | tabby-and-white | currently fattest cat, food-dish camper, lazy | xl | bedroom | Cole |
| **Serval** | tabby-and-white | senior, thinning, was XL | large | office | Cole |
| **Summer** | cream | smart, demanding, anxious-with-strangers, longer coat | medium dog | bedroom / living_room / outdoor | Cole |
| **Dalila** | brindle | low-energy, chill, "Eeyore" | medium dog | bedroom / living_room / outdoor | Cole |

**Sneaky is non-monolocational** — she cycles between living-room chairs, the kitchen island, and various closets. Rather than forcing a single `home_room`, her config uses `home_room: cyclic` and a `cyclic_home_rooms` list. The world model treats her `IN_HOUSE_UNMONITORED` reasoning as *"probably one of {living_room, kitchen, a closet}, last seen X minutes ago"* — honest hedging instead of false specificity.

**Three discrimination tiers:**

**Tier 1 — Trivial (≥0.85 confidence on visual signals alone):** Socks (tuxedo, unique), Sneaky (silver-tabby + blue eyes, unique), Summer vs Dalila (cream-longhair vs brindle-shorthair, different breed-class). The two dogs are also trivially distinguishable from any cat at any reasonable resolution.

**Tier 2 — Behavior-and-size discrimination (Sparta vs Serval).** Both tabby-and-white. The discriminating signal is **dynamic size + behavioral priors**.

- **Static size is wrong.** Sparta is currently obese (will diet someday). Serval is senior-thinning from a former XL frame. Either may shift over months. Static config values would degrade silently.
- **Dynamic size** is correct: per-entity rolling-median bbox area over the last 200 confident-attribution observations. The cost function reads from the rolling estimate; observations contribute to whichever cat the *non-size* signals identified (location, behavior); size becomes self-correcting. Cold-start uses the static `expected_size` until enough history accumulates (default: 7 days).
- **Behavioral priors do most of the work.** Sparta camps the food dish (see §22.9 landmark `food_dish` interaction frequency). Serval's home_room is `office`, Sparta's is `bedroom`. Sparta is lazy (low motion variance), Serval is senior (low overall activity, more punctuated). Combined with size, attribution is reliable.

When confidence falls below 0.6, the persona hedges explicitly: *"I see a tabby-and-white in the bedroom — likely Sparta given the location, but it could be Serval if he's wandered in."* Wrong-confident is worse than hedged.

**Tier 2b — Coat-texture and location discrimination (Spooky vs Velcro).** Both solid black. The discriminating signals are **location prior + coat texture + behavioral signature**.

- **Location prior is the strongest signal.** Velcro's home is unmonitored (Jeff's room). The system sees Velcro primarily through *transitions* into monitored space:
  - Black cat appears from a `jeff_room` exit polygon → strong Velcro prior.
  - Black cat on top of the kitchen fridge → strong Velcro prior (preferred-perch landmark).
  - Black cat asleep on the bed at night next to Cole → strong Spooky prior (sleep_partner relationship).
  - Black cat hissing at or fleeing Summer → strong Velcro prior (`conflicts_with: [summer]`); Spooky is calm around the dogs.
- **Coat texture as a real classifier.** Spooky is fluffy/curly; Velcro is straight/sleek. The §22.3 enricher computes a coat-texture descriptor from the bbox crop using **edge-density variance** (fluffy = high-variance silhouette edges) plus **Gabor-filter response statistics** at multiple scales. CLIP zero-shot with prompts `"a fluffy curly cat"` vs `"a sleek straight-coated cat"` is the lazier alternative; accuracy is fine for the binary problem at typical Wyze distances.
- **The "Velcro is in Jeff's room" default.** Velcro's `unmonitored_home: jeff_room` lets the system confidently report `IN_HOUSE_UNMONITORED, last_seen=jeff_room` whenever direct observation is stale. *"Where's Velcro?"* → *"Jeff's room, probably."*
- **Velcro vs Summer interaction.** Velcro hates Summer. A black cat fleeing/hissing at Summer is almost certainly Velcro; a black cat near Summer without distress signals leans Spooky. The interaction inference module (§24) tags these as `conflict_interaction` events; the cost function reads them as discrimination signals.
- **Sneaky's hyperawareness of Summer** is the *inverse* signal. Sneaky pre-positions toward exits when Summer is nearby — defensive surveillance, not active conflict. Different observable signature, same input (Summer's location), useful for distinguishing Sneaky's behavior from any other cat's.

**Tier 3 — Doesn't exist.** No triple of visually identical animals. Every hard case is a pair; every pair has at least one strong non-visual signal.

**Decorative tag — Sneaky sees ghosts.** `sees_ghosts: true` in config. No functional effect on the world model. But when the persona notices Sneaky staring at a wall and someone asks why, it has the right tone available: *"Sneaky's seeing things again."* Costs nothing, makes the persona feel like it knows your cats.

**Summary of which signals do which work:**

| Pair | Primary | Secondary | Tertiary |
|---|---|---|---|
| Socks vs anyone | color (tuxedo) | — | — |
| Sneaky vs anyone | color (silver-tabby + blue eyes) | — | — |
| Sparta vs Serval | dynamic size | landmark prior (food_dish) + room prior | motion features |
| Spooky vs Velcro | location prior (transitions, fridge, sleep-partner) | coat texture | Summer-interaction signature |
| Summer vs Dalila | color + breed_class | — | — |
| Cats vs dogs | YOLO species class | — | — |
```

---

## EDIT 5 — §22.2 Pet declaration in config (FINAL)

**Anchor:** §22.2, full section.

**REPLACE WITH:**

```markdown
### 22.2 Pet declaration in config

```yaml
# config.yaml

world_model:
  tracked_species: [cat, dog]
  visiting_animal_retention_minutes: 60

residents:
  - id: cole
    display_name: Cole
    primary_room: office
  - id: anna
    display_name: Anna
    primary_room: bedroom
  - id: jeff
    display_name: Jeff
    primary_room: jeff_room          # declared but UNMONITORED

pets:
  cats:
    - name: Socks
      household_owner: cole
      color_class: tuxedo
      expected_size: medium
      home_room: office              # ~99% of time in the office
      personality: skittish_weird
      affinities:
        - person: anna
          strength: high
          contexts: [proximity_general]

    - name: Sneaky
      household_owner: cole
      color_class: silver-tabby
      coat_length: long
      expected_size: medium-large
      home_room: cyclic
      cyclic_home_rooms: [living_room, kitchen, any_closet]
      personality: patient_hunter_survivalist
      hyper_alert_to: [summer]
      sees_ghosts: true
      distinctive_features: [blue_eyes, pink_nose, long_coat]

    - name: Spooky
      household_owner: cole
      color_class: black
      coat_texture: fluffy_curly
      expected_size: medium
      home_room: bedroom
      personality: calm
      affinities:
        - person: cole
          strength: high
          contexts: [sleeping]

    - name: Velcro
      household_owner: jeff
      unmonitored_home: jeff_room
      color_class: black
      coat_texture: straight_sleek
      expected_size: medium-large    # bigger than Spooky
      home_room: jeff_room
      personality: spirited
      preferred_perches: [kitchen_fridge_top]
      conflicts_with: [summer]
      affinities:
        - person: jeff
          strength: high
          contexts: [proximity_general, sleeping]
        - person: anna
          strength: medium
          contexts: [physical_contact, rubbing]

    - name: Sparta
      household_owner: cole
      color_class: tabby-and-white
      expected_size: xl
      size_basis: dynamic
      home_room: bedroom
      personality: lazy_opportunist
      preferred_landmarks: [food_dish]
      affinities:
        - person: anna
          strength: medium
          contexts: [proximity_general]
      notes: "Affinity is soft — Sparta gravitates to whoever is most convenient."

    - name: Serval
      household_owner: cole
      color_class: tabby-and-white
      expected_size: large
      size_basis: dynamic
      home_room: office
      personality: senior
      age_state: senior_thinning

  dogs:
    - name: Summer
      household_owner: cole
      breed_class: medium-longhair
      color_class: cream
      expected_size: medium
      home_rooms: [bedroom, living_room, outdoor]
      feeding_room: living_room
      personality: smart_demanding_excitable
      anxiety_triggers: [strangers_in_house]
      affinities:
        - person: anna
          strength: high
          contexts: [proximity_general]
        - person: cole
          strength: medium
          contexts: [authority]

    - name: Dalila
      household_owner: cole
      breed_class: medium-shorthair
      color_class: brindle
      expected_size: medium
      home_rooms: [bedroom, living_room, outdoor]
      feeding_room: living_room
      personality: low_energy_chill
      nicknames: [Eeyore, brown_dog, dog_2]

# Out-of-scope for the world model. See §22.10.
outdoor_pets:
  - name: Scooter
    species: tortoise
    breed: sulcata
    typical_zones: [backyard]
    household_owner: cole
```
```

---

## EDIT 6 — §22.3 Animal enrichment in ObservationBuilder

**Anchor:** §22.3 heading and intro paragraph (before the original `_build_cat_obs` code).

**REPLACE the heading and intro WITH:**

```markdown
### 22.3 Animal enrichment in ObservationBuilder

Detections in `tracked_species` flow through `_build_animal_obs(species, ...)`, which dispatches to a per-species enricher. Cats use color/size/coat-texture; dogs use color/size/breed-class.

**Dispatcher:**

```python
_ENRICHERS = {"cat": "_build_cat_obs", "dog": "_build_dog_obs"}

def _build_animal_obs(self, species, frame, det, room, ts, fw, fh):
    method_name = self._ENRICHERS.get(species)
    if method_name is None:
        return self._build_generic_animal_obs(species, frame, det, room, ts, fw, fh)
    return getattr(self, method_name)(frame, det, room, ts, fw, fh)
```

**Coat-texture descriptor (cats only, used for Spooky/Velcro disambiguation):** computed on the bbox crop. Two complementary signals:

1. **Edge-density variance.** Run a Sobel filter on the silhouette band (just inside the bbox edge); compute the variance of edge-magnitude along the perimeter. Fluffy/curly coats produce high-variance edge profiles; sleek coats produce low-variance. Output: scalar in [0, 1].
2. **Gabor-filter response statistics.** Apply Gabor filters at 4 orientations × 3 scales to the bbox interior; compute mean and variance of responses. Different fur textures produce different frequency-domain signatures. Output: 24-dim vector, reduced to 4-dim via per-species PCA at training time.

The two signals concatenate to a 5-dim coat-texture vector that becomes part of the cat observation. CLIP zero-shot is the lazier alternative if classical-vision implementation feels heavy; accuracy at the binary fluffy-vs-sleek problem is fine on Wyze cameras.

**Dog-specific enrichment (`_build_dog_obs`):**

- **Color classifier** (8 classes): tan, brown, black, white, tricolor, merle, brindle, cream.
- **Breed-class classifier** (7 classes): small-shorthair, small-longhair, medium-shorthair, medium-longhair, large-shorthair, large-longhair, giant. *Not* breed identification — a stable visual class that any household with size/coat-distinct dogs can use for re-id. Implementation: CLIP zero-shot with fixed prompt set is the recommended default.

The cat enrichment code from the original §22.3 follows below (unchanged), with `_build_dog_obs` added alongside.
```

(The existing `_build_cat_obs` code block stays as written.)

---

## EDIT 7 — §22.7 cost function (light edit)

**Anchor:** §22.7 heading.

**REPLACE the heading line WITH:**

```markdown
### 22.7 The animal cost function (cats shown; dogs analogous)
```

**APPEND at the end of §22.7:**

```markdown
**For dogs**, structurally identical. Substitute weights: lower `w_room_prior` by ~30% (dogs follow humans across rooms more), raise `w_size` by ~50% in size-disparate dog households (Summer-vs-Dalila is essentially a free win on size + breed_class alone). For same-size same-coat dogs (the Smudge/Onyx-equivalent for dogs, which this household does not have), the same hedging discipline as cats applies — confidence below 0.6 produces explicitly hedged persona output.
```

---

## EDIT 8 — §22.9 species-specific events

**Anchor:** §22.9 heading.

**REPLACE the heading WITH:**

```markdown
### 22.9 Species-specific events worth firing
```

**APPEND at the end of §22.9** (before the section terminator):

```markdown
**Dog-specific landmarks and events:**

```yaml
rooms:
  - id: living_room
    world_model:
      landmarks:
        - name: dog_food_dish
          polygon: [[450, 500], [580, 500], [580, 600], [450, 600]]
        - name: dog_water_bowl
          polygon: [[600, 500], [720, 500], [720, 600], [600, 600]]
        - name: leash_hook
          polygon: [[50, 100], [150, 100], [150, 200], [50, 200]]
```

Producing `DOG_FOOD_VISIT`, `DOG_WATER_VISIT`, `LEASH_INTERACTION` events. `LEASH_INTERACTION` is a human + dog co-located on `leash_hook` for ≥3s — strong predictor of imminent walk; the persona can preemptively note *"Looks like Summer's about to get a walk."*

If a dog door is ever installed, its polygon resolves disappearance the same way an `exterior_exit` polygon does for humans — see §22.0 and §29.5. *Not currently applicable* to this household.
```

---

## EDIT 9 — §22.10 Outdoor-only animals (NEW)

**Anchor:** insert after §22.9.

**INSERT:**

```markdown
### 22.10 Outdoor-only animals (Scooter / OutdoorObserver)

Scooter is a sulcata tortoise. He lives outdoors. The world model **does not track Scooter** by default; he is handled by a separate, lightweight `OutdoorObserver` subsystem.

**Why he doesn't fit the world model:**

1. YOLOv8 doesn't detect tortoise as a class. Detection requires either a finetuned head with hand-labeled tortoise data, or an open-vocab detector (OWL-ViT / GroundingDINO) prompted with `"a tortoise"`. Open-vocab is the lower-cost path.
2. The bounded-house model breaks: Scooter's default state is *outside*, and his disappearance is explained by going under a bush, not by crossing a door polygon.
3. Re-id is trivial — there's exactly one tortoise. The disambiguation machinery from §22 is wasted on him.
4. Query value is narrow.

**OutdoorObserver design** (separate module, not part of WorldModel):

- One outdoor camera with a defined "tortoise zone" polygon.
- OWL-ViT at 1 fps on motion-triggered frames only.
- Single entity, no re-id, no behavioral profile. Tracks `last_seen_outdoor` timestamp and zone.
- Persona tool: `where_is_scooter()` → timestamp + freshness bucket (recent / today / multiple days).
- Reusable scaffold: same module pattern handles *"has anyone been in the backyard?"* and *"did the package get delivered?"*

**Crucial semantic distinction.** The token `outdoor` in the *dog* config (`home_rooms: [bedroom, living_room, outdoor]`) means "the dogs go in the backyard sometimes, that's normal, no alarm." The token `outdoor` in any *cat* context is **never normal** and triggers §29's cat-escape alarm. The §22.10 OutdoorObserver and §29's cat-escape alarm subscribe to *different* outdoor-zone polygons:

- Dogs in the backyard zone → OutdoorObserver records the visit, no alarm.
- Cats anywhere outside an interior `exterior_exit` polygon → cat-escape alarm fires.

These polygons must not overlap or accidentally share semantics. v4 enforces this in schema validation: an `exterior_exit` polygon and an `outdoor_zone` polygon cannot share coordinates.

**Recommendation:** ship without Scooter for the demo. Build OutdoorObserver as Phase 6 if it earns its keep alongside other backyard-monitoring use cases.
```

---

## EDIT 10 — §22.11 Adding a new pet (NEW)

**Anchor:** insert after §22.10.

**INSERT:**

```markdown
### 22.11 Adding a new pet (workflow)

1. **Edit `config.yaml`**: add the pet under `pets.<species>:` with all fields. If species is new, add to `world_model.tracked_species`.
2. **Take 5 enrollment photos** via the dashboard's *Enroll Pet* page. The page captures crops, saves to `pet_samples/<species>/<name>/`, calls `enroll_animal(...)` with the populated config metadata, and shows a confidence rating (re-identifying the same crops). If any of 5 mis-match, prompts for a 6th.
3. **Hot-reload `pets.*` blocks** without restart. The `PetsManager` re-bootstraps. The `tracked_species` list itself requires restart (gates pipeline wiring at startup); the dashboard shows "restart required" warning when species list changes.
4. **Verify in the polygon viewer.** New pet should show as `state=PRESENT` with the right name within ~5 minutes of camera exposure.
5. **Behavioral profile builds passively** over `min_history_days` (default 14). During that window, anomaly scoring is skipped for the new pet.

**Removing a pet** (rehomed, passed away, roommate moves out): set `archived: true` in config or remove the entry. The `PetsManager` flips `archived_at = now()`. Row stays for history queries; new observations don't re-link to archived entities.

**Add-a-pet is never an auto-promotion path.** Frequent visitors do not auto-become residents. See §22.0.
```

---

## EDIT 11 — §13 Human state machine: PRESENT_OFF_FRAME (NEW state)

**Anchor:** §13, the human state machine definition.

**INSERT a new subsection §13.x after the existing state machine definition:**

```markdown
### 13.x The PRESENT_OFF_FRAME state

The original state machine has PRESENT / IN_ROOM_UNSEEN / IN_HOUSE_UNMONITORED / DEPARTED. v4 adds **PRESENT_OFF_FRAME**, used during transitions where a human is partially or recently out of camera frame but is still presumed in or near the room.

**Trigger conditions:**

- Human bbox was visible in frame within the last `T_off_frame_grace_seconds` (default: 8s).
- Bbox is currently absent OR is significantly clipped by the frame edge (≥40% of bbox area would be outside the frame if extrapolated).
- The trajectory of the bbox before clipping/disappearance was *toward* the frame edge or toward a known door polygon.

**State transitions involving PRESENT_OFF_FRAME:**

```
PRESENT --(trajectory toward edge + clipping or disappearance)--> PRESENT_OFF_FRAME
PRESENT_OFF_FRAME --(reappears in same camera within grace)--> PRESENT
PRESENT_OFF_FRAME --(trajectory was toward exterior_exit polygon)--> EXITED_VIA_DOOR(door_id)
PRESENT_OFF_FRAME --(grace expires, no reappearance, not toward known exit)--> IN_ROOM_UNSEEN
EXITED_VIA_DOOR --(reappears in any camera within T_door_return_seconds)--> PRESENT
EXITED_VIA_DOOR --(T_door_return_seconds expires)--> DEPARTED
```

**Tunables:** `T_off_frame_grace_seconds` (default 8), `T_door_return_seconds` (default 60 — covers "stepped onto the porch to grab a package").

**Bbox clip-fraction calculation:**

```python
def bbox_clip_fraction(bbox: BBox, frame_w: int, frame_h: int) -> float:
    """Fraction of bbox edges touching frame edges. High value = partial observation."""
    edges_clipped = 0
    if bbox.x0 <= 1: edges_clipped += 1
    if bbox.y0 <= 1: edges_clipped += 1
    if bbox.x1 >= frame_w - 1: edges_clipped += 1
    if bbox.y1 >= frame_h - 1: edges_clipped += 1
    return edges_clipped / 4.0
```

A clip-fraction of 0.5 (two edges touching frame edges) is a strong PRESENT_OFF_FRAME signal — likely the person is in a doorway with their leg/half-body in frame. The state machine biases toward PRESENT_OFF_FRAME during clipping rather than jumping to IN_ROOM_UNSEEN, which would be wrong.

**Why this matters beyond the door alarm.** PRESENT_OFF_FRAME improves "where's Cole?" answers across the system: *"right outside the back door, came in 30 seconds ago"* is a much better answer than *"unknown, last seen kitchen 35 seconds ago."* The Phase 3 demo benefits directly. The door-open-alarm correctness in §29 is the most visible application but not the only one.
```

---

## EDIT 12 — §29 Alarm Subsystem (NEW TOP-LEVEL SECTION)

**Anchor:** insert as new top-level section after §28.

**INSERT:**

```markdown
## 29. Alarm Subsystem

A multi-alarm framework for hard-rule safety alerts. Three alarm types in v4: **Fire Detected**, **Cat Escape**, **Door Open Without Human**. Architecturally extensible — adding a fourth alarm type is a new subscriber to the same framework.

This is *not* part of the anomaly scorer. Anomalies are statistical and tunable; alarms are deterministic hard rules with sub-second latency and asymmetric error costs (false-positive = mild annoyance; false-negative = serious harm). Different tools.

### 29.1 The framework

**Module:** `modules/safety/alarms/`. Each alarm type is its own subscriber, sharing infrastructure:

- `AlarmAudio` — speaker fan-out, klaxon + TTS mixing, ducking, escalation cadence.
- `AlarmState` — persisted state machine per alarm (ACTIVE / FIRING_AUDIO / MUTED / RESOLVED / SUPPRESSED).
- `AlarmPriority` — the concurrent-audio policy (§29.6).
- `AlarmConditionWatch` — subscribes to underlying physical-condition signals during FIRING_AUDIO so condition-clearance can resolve the alarm immediately.

**Alarm state machine** (per alarm instance):

```
INACTIVE
  --(condition true, not suppressed)--> FIRING_AUDIO
  
FIRING_AUDIO
  --(condition false)--> RESOLVED                    [primary auto-resolve]
  --(voice silence)--> MUTED                         [secondary, with rearm]
  --(visual confirmation, where applicable)--> MUTED [secondary, with rearm]
  
MUTED
  --(condition false)--> RESOLVED
  --(mute timer expires AND condition still true)--> FIRING_AUDIO  [rearm]
  --(escalation override condition met)--> FIRING_AUDIO            [hard override]
  
RESOLVED
  --(condition true again)--> FIRING_AUDIO           [retrigger]
```

**Condition-clearance is the primary resolve path for every alarm.** The alarm subscribes to its underlying condition signal (cat outside? door open? fire detected?) *while firing*, not just to trigger the fire. The moment the condition clears, the alarm resolves on its own — no voice command needed, no inference about human awareness needed. Audio stops within 1-2 seconds of the condition going false (bounded by detection-rate latency).

**Why this matters for UX.** The "door closes at second 16, alarm fires belatedly at second 18" case resolves correctly: at second 19-20, the alarm sees door=closed, condition has cleared, audio stops. Total alarm duration: 2-3 seconds. The system isn't perfect-prediction; it's perfect-correction.

**Voice silence is secondary.** It exists for the case where the audio is bothering you while the condition is still active and you want to stop it for a window. Not the same as condition-clearance.

### 29.2 Cat-Escape Alarm

**Trigger condition:** observation has `entity_type='cat'` AND bbox center is inside an `exterior_exit` polygon AND alarm is armed AND no per-cat suppression active.

**Firing latency budget:** ~300ms total (detection → enrichment → bus → subscriber → fan-out). See v3 for the breakdown; unchanged in v4.

**Audio:** klaxon (cat-escape klaxon, distinct from fire/door — see §29.5) + TTS shouting `"[EXIT_NAME]. [CAT_NAME] IS OUTSIDE."` Repeats with klaxon between announcements. Escalates after 30s (faster cadence, added high-frequency overtone in klaxon).

Examples:
- *"[KLAXON] BACK DOOR. SNEAKY IS OUTSIDE."*
- *"[KLAXON] FRONT DOOR. UNIDENTIFIED CAT IS OUTSIDE."*

`exit_name` comes from a new `display_name` field on each `exterior_exit` polygon (e.g., id=`back_door_exit`, display_name=`BACK DOOR`).

**Auto-resolve (primary):** the cat is observed in any monitored interior room. World model publishes `entity.state_changed` with `new_state=PRESENT`; alarm subscribes during FIRING_AUDIO. Cat back inside → audio stops in 1-2s.

**Suppression mechanisms** (all from v3, unchanged):

1. **Global disarm** (`armed=false` for some duration). Voice: *"Jarvis, disarm cat escape alarm for [duration]."* Auto-rearms.
2. **Per-cat suppression** (planned outing): *"Jarvis, I'm taking Sneaky out for 20 minutes."* Other cats remain armed.
3. **Post-fire mute** (5min default). Voice command silences current audio with rearm if condition persists.

**Intentional-retrieval window:** if a human exits through the same door the cat exited through within 30s of the alarm firing, the alarm enters a 5-min mute for that cat (the human is presumed to be retrieving). Tunable: `cat_escape_intentional_retrieval_window_seconds`, default 30. Outside the window, the same human exit doesn't trigger this — at 5 PM the system shouldn't infer "Cole is retrieving the cat from this morning's escape."

**Pre-arm check** at noon daily (per Cole's preference). Synthesizes a test observation per `exterior_exit` polygon, verifies subscriber path and speaker reachability, plays a soft chime confirmation.

### 29.3 Door-Open-Without-Human Alarm

**Trigger condition:** door is detected open AND no human is observed within `door_supervision_radius_m` (default 3m) of the door for `door_unsupervised_grace_seconds` (default 15s).

**"Within 3m of door"** semantics: the human's bbox center, projected to room coordinates, is within 3m of the door polygon's center. **If no human is visible to the camera at all, treat as >3m away.** This biases toward firing when uncertain — correct asymmetry.

**(b) suppression-while-near logic:** human within 3m → countdown does not start. Human leaves the 3m radius (or off-frame) with door still open → 15s countdown begins. Door closes during countdown → countdown cancels.

**Door state detection — vision-primary, reed-switch retrofit-ready:**

```python
# modules/safety/alarms/door_state_vision.py
class VisionDoorMonitor:
    """
    Detects door state from camera frames.
    Uses geometric pose detection on known door regions.
    Publishes 'door.state' events to the bus.
    """
    ...

# modules/safety/alarms/door_state_reed.py
# class ReedSwitchDoorMonitor:
#     """
#     STUB — uncomment and wire when reed switches are installed.
#     Hardware-backed door state via reed switches on perimeter doors.
#     Massively more reliable than vision; recommended retrofit when bandwidth allows.
#     Wire each reed to an ESP32 GPIO; ESP32 publishes 'door.state' events to the
#     same bus topic. The DoorOpenAlarm subsystem listens to 'door.state' identically
#     whether events originate from vision or reed — keep schema stable.
#     """
#     def __init__(self, gpio_map: dict[str, int], esp32_node: ESP32Client):
#         ...
```

The alarm subscribes to `door.state` events without caring about the source. Future reed retrofit changes the publisher only.

**Trajectory-aware door logic** (§13.x):

- Human walks to door, walks through, transitions to PRESENT_OFF_FRAME → EXITED_VIA_DOOR.
- Door is open. Door alarm subscribes to door state and to recent EXITED_VIA_DOOR events.
- If door was opened by a recent human exit through it → countdown does NOT start (the human is on the porch / just outside).
- Door alarm starts countdown only if door is open AND no recent EXITED_VIA_DOOR for this door.
- If the human returns within `T_door_return_seconds` (default 60), state transitions back to PRESENT, no alarm path engages.
- If the human stays outside past `T_door_return_seconds` → state transitions to DEPARTED. Door alarm engages only if door is *still open* at that point — e.g., they walked away and forgot to close it.

**Audio:** door-open klaxon (distinct, lower urgency — see §29.5) + TTS `"[DOOR_NAME] OPEN."` Long inter-pulse gaps. Escalation = shorter gaps, no frequency change.

**Auto-resolve (primary):** door is closed. Door state subscriber sees state=closed → audio stops in 1-2s. *This is the second-18 case.*

**Camera-FOV-edge note.** A human in a doorway with most of their body off-frame produces a partial bbox observation, which §13.x's PRESENT_OFF_FRAME logic correctly handles — the system understands they're still there. Failure mode: the human is *fully* outside the camera FOV, near the door, but not visible at all (camera's FOV doesn't cover the porch). In that case the system treats them as absent and the countdown engages. *Annoying but recoverable*: human says *"Jarvis, stop alarm"* when they hear it. Best fix is camera placement (wider FOV at exits), not algorithmic.

### 29.4 Fire-Detected Alarm

**Trigger condition:** fire detection signal active (smoke / thermal / vision-based fire detection).

**Audio:** fire klaxon (low-frequency horn, slow pulse, designed to penetrate walls and travel through smoke) + TTS `"FIRE DETECTED IN [ROOM]."`

**Resident-trust model.** This household has competent adults and a commercial-grade fire extinguisher. The auto-resolve logic accordingly trusts that residents who see the fire are handling it, and treats the alarm as informational once it's been acknowledged. *Sustained klaxon while you're holding an extinguisher creates more risk than it mitigates.*

**Auto-resolve paths** (any of the following):

1. **Condition-clearance (primary).** Fire detection signal clears for `fire_signal_clearance_seconds` (default 60s) of continuous absence. Alarm resolves naturally.
2. **Voice silence.** *"Jarvis, I see it"* / *"Jarvis, I'm on it"* / *"Jarvis, stop alarm."* Enters 3-minute MUTED state. Audio stops. **Phone alerts continue normally** — see §29.6.
3. **Visual confirmation with non-increasing signal.** Human enters fire room AND remains for ≥`fire_visual_confirmation_dwell_seconds` (default 3s) AND gaze/orientation toward fire region OR moving toward it OR within 2m of it AND fire detection signal is *not increasing*. Enters 3-minute MUTED state.

**The non-increasing-signal condition is critical.** If a human enters the fire room and the fire is *growing*, the alarm does NOT silence — they may not have seen it yet, they may be assessing, or they may be losing the fight. Audio continues until either condition-clearance, voice-silence, or signal-stops-growing-for-N-seconds.

**Rearm conditions during MUTED:**

- **Signal-increase override** (hardest rearm): if detection signal increases meaningfully after silence, alarm re-fires immediately, *cannot be silenced again for 30 seconds*. The fire is winning; the audio comes back regardless of human acknowledgment.
- **Unattended-rearm.** If detection signal is still active after `fire_alarm_unattended_rearm_seconds` (default 300s = 5min) of silence, AND no human is currently in the fire room, alarm re-fires. Covers the case: saw the fire, went to grab extinguisher, got distracted, fire still burning. Tunable to 0 to disable.
- **Mute-timer expiration.** Default 3min. If condition still active and no override fired, audio resumes.

**Broadcast on silence.** When the alarm is silenced (any mechanism), a notification fires to *every* household member's phone via §31's dispatcher: *"Cole acknowledged fire alarm in kitchen at 14:14."* Other residents need to know a fire was detected even if the present cohabitant handled it. The audio silence is local-room behavioral nudge; the broadcast is the distributed record.

**Where this differs from typical "smart home" fire alarms.** Most consumer systems can't be silenced without a code, on the theory that any silencing is dangerous. This design accepts more silencing risk in exchange for better real-firefighting ergonomics, with the understanding that:

(a) the residents are competent and trained,
(b) the broadcast-on-silence ensures distributed awareness,
(c) the unattended-rearm catches distracted-resident cases,
(d) the signal-increase override catches losing-the-fight cases.

If the household composition changes (kids, frequent guests, less-trained residents), revisit this design. The trust model is the load-bearing assumption.

### 29.5 Distinct klaxons per alarm

| Alarm | Base klaxon | Escalation (after 30s) | Rationale |
|---|---|---|---|
| **Fire** | Continuous low-frequency horn (~150Hz), slow pulse | Faster pulse + added high-frequency overtone | Low frequency penetrates walls and smoke. Overtone is psychoacoustically more urgent without painful. |
| **Cat escape** | Three-note rising chirp (~600/800/1000 Hz), repeating | Compressed inter-repeat interval, added fourth note up top | Distinct frequency band and pattern. Alerting without industrial. |
| **Door open** | Single mid-frequency tone (~440Hz), short pulse, long gap | Shorter inter-pulse gaps, no frequency change | Lowest urgency. Long gaps signal informational. Shorter gaps signal "address this." |

**Audio file structure:**

```
assets/alarms/
  fire_base.wav             # 4s loop
  fire_escalated.wav        # 4s loop
  cat_escape_base.wav       # 3s loop
  cat_escape_escalated.wav  # 3s loop
  door_open_base.wav        # 8s loop with long gap
  door_open_escalated.wav   # 8s loop, shorter gap
  
  tts_announcements/        # generated on-demand by Piper, cached
    .gitignore
```

**File requirements:** 48kHz / 16-bit / mono / WAV / loudness-normalized to -14 LUFS. The TTS announcement plays *over* the klaxon at lower klaxon volume (ducking: klaxon × 0.4 during announcement, × 1.0 between). v4 ships with placeholder WAVs labeled `REPLACE_ME_*` so the system functions before real audio is produced. Generate real audio via ElevenLabs sound effects, FreeSound + DAW, or hand-recording.

### 29.6 Concurrent audio: priority and policy

When multiple alarms are simultaneously active, the audio doesn't stack — that produces unintelligible noise. The audio is held by the highest-priority active alarm; lower-priority alarms remain ACTIVE in the subsystem (logged, in dashboard, sending phone alerts) but are AUDIO-SUPPRESSED.

**Priority order (highest first):**

1. **Fire detected** — life safety, highest.
2. **Cat escape** — pet safety, high.
3. **Door open without human** — perimeter integrity, lower.

**Concurrent audio policy:**

- Highest-priority active alarm owns the speakers.
- Lower-priority alarms are AUDIO-SUPPRESSED but ACTIVE.
- TTS announcement adds a **suffix** when other alarms are also active, once per loop cycle (not every announcement repetition):
  - *"BACK DOOR. SNEAKY IS OUTSIDE. ALSO: DOOR OPEN ALARM ACTIVE."*
- When the holding alarm resolves, the next-highest-priority active alarm immediately reclaims the audio.
- Phone alerts via §31 fire **per alarm, independently**. No merging, no suppression. Three concurrent alarms = three separate notifications.

**Voice-silence semantics with multiple alarms:**

- *"Jarvis, stop alarm"* silences whichever alarm currently holds the audio. That alarm enters MUTED. The next-highest-priority active alarm immediately takes the audio. So if cat-escape and door-open are both active, "stop alarm" silences cat-escape's audio (its mute timer starts) and door-open's audio begins.
- *"Jarvis, silence all alarms"* mutes every currently-audible alarm with their respective rearm timers. Use sparingly. The dashboard banners and phone notifications continue.
- The **fire alarm signal-increase override** (§29.4) cannot be silenced by either command for 30s. Hardware-level priority for the failure mode that matters most.

**Real escape scenario walk-through.** Door blows open from wind, no human present (door alarm starts countdown). Velcro slips through during second 8 (cat alarm fires immediately at second 8, takes the audio over the not-yet-firing door alarm — cat alarm > door alarm priority). Door alarm fires at second 15 (door still open, no human within 3m); door alarm is AUDIO-SUPPRESSED but ACTIVE. TTS: *"FRONT DOOR. VELCRO IS OUTSIDE. ALSO: DOOR OPEN ALARM ACTIVE."* Cole hears it, runs to the front door, retrieves Velcro, brings him in, closes the door. Cat alarm condition clears (Velcro back inside) → cat alarm resolves. Door alarm condition clears (door closed) → door alarm resolves. Both phone alerts had already gone out. Total system-engaged duration: maybe 90 seconds. Both auto-resolved without a single voice command.

### 29.7 Implementation skeleton (multi-alarm)

```python
# modules/safety/alarms/dispatcher.py

class AlarmDispatcher:
    def __init__(self, audio: AlarmAudio, persistence: Persistence):
        self.alarms: dict[str, Alarm] = {}
        self.audio = audio
        self.persistence = persistence
        self.priority_order = ["fire", "cat_escape", "door_open"]

    def register(self, name: str, alarm: Alarm):
        self.alarms[name] = alarm

    def active_alarms(self) -> list[str]:
        return [n for n in self.priority_order
                if self.alarms[n].state in ("FIRING_AUDIO", "MUTED")]

    def audio_owner(self) -> str | None:
        """Highest-priority alarm currently in FIRING_AUDIO."""
        for name in self.priority_order:
            if self.alarms[name].state == "FIRING_AUDIO":
                return name
        return None

    async def on_alarm_state_change(self, name: str, old: str, new: str):
        # Reclaim or release audio.
        owner = self.audio_owner()
        if owner is None:
            await self.audio.stop()
            return
        if owner == name and new == "FIRING_AUDIO":
            await self.audio.play_for(name, suffix=self._suffix())
        elif owner != name and old == "FIRING_AUDIO":
            await self.audio.play_for(owner, suffix=self._suffix())

    def _suffix(self) -> str:
        active = self.active_alarms()
        if len(active) <= 1:
            return ""
        owner = self.audio_owner()
        others = [a.upper().replace("_", " ") for a in active if a != owner]
        return f" ALSO: {', '.join(others)} ALARM ACTIVE."

    async def voice_silence_current(self):
        owner = self.audio_owner()
        if owner is None:
            return
        await self.alarms[owner].voice_silence()

    async def voice_silence_all(self):
        for name in self.priority_order:
            if self.alarms[name].state == "FIRING_AUDIO":
                await self.alarms[name].voice_silence()
```

The per-alarm `Alarm` subclasses (FireAlarm, CatEscapeAlarm, DoorOpenAlarm) implement their own condition watch, voice-silence handlers, and rearm logic. The dispatcher only owns audio routing and suffix construction. Clean separation; each alarm is auditable in isolation.
```

---

## EDIT 13 — §30 Wake words and personas (NEW TOP-LEVEL SECTION)

**Anchor:** insert after §29.

**INSERT:**

```markdown
## 30. Wake Words and Personas

The system has three wake words: **Jarvis** (primary), **Mira**, and **GLaDOS**. Each routes to a distinct persona with its own system prompt and voice, but shares the underlying world model, tools, memory, and alarm subsystem. The persona choice biases response *style*, not capability.

### 30.1 Why three wake words and not one

Same orchestrator, three personality skins. *"Jarvis, status report"* gets a clipped technical readout. *"GLaDOS, status report"* gets the same data with deadpan commentary. *"Mira, what's going on?"* gets a softer narrative. The information underneath is identical; only the delivery differs.

This is cheap to implement (three system prompt files, one routing function, three voice configs) and useful in practice: different moods and contexts call for different tones, and having the wake word itself be the mood selector is a much better UX than asking the system to "be more technical please."

**Important — Mira-the-Telegram-bot stays separate.** The existing Mira project is a Telegram companion bot with its own runtime, memory, and FLUX/Ollama backend. The Jarvis-Mira-persona shares the *name* with the Telegram Mira but is a different process with different memory. If the two ever need to share memory, the right architecture is a shared memory store both query, not a merged process. Future problem; flagging now to avoid accidental coupling.

### 30.2 Schema

```yaml
wake_words:
  - word: jarvis
    persona: jarvis_default
    primary: true
  - word: mira
    persona: mira_warm
  - word: glados
    persona: glados_dry

personas:
  jarvis_default:
    system_prompt_file: prompts/jarvis_system.md
    voice: en_US-amy-medium       # Piper voice id
    style_hint: technical_concise
  mira_warm:
    system_prompt_file: prompts/mira_system.md
    voice: en_US-libritts-high
    style_hint: warm_narrative
  glados_dry:
    system_prompt_file: prompts/glados_system.md
    voice: en_US-glados-custom    # or any other voice; finetuning optional
    style_hint: deadpan_sardonic
```

### 30.3 Wake-word detection: name-only with self-suppression

The system supports **name-only wake**: *"Jarvis."* (pause) *"What's the weather."* No "hey" prefix required. This trades a small false-wake risk for substantially better UX.

**Self-suppression layers** that mitigate false wakes:

1. **Own-speech mute.** Wake detection is suppressed while the system is itself speaking, AND for `T_post_speech_suppression_seconds` (default 2s) after speech ends. Catches the case where the system says its own name in a response and would otherwise wake itself.

2. **Post-detection addressivity check.** When a wake word fires, the system listens for `T_addressivity_window_seconds` (default 1s) before committing to a wake. If the audio after the wake word is silence, continued conversation between humans, or doesn't parse as a query/command, the wake is aborted silently (no chime, no light, no log entry beyond a debug-level event). Catches *"Jarvis is being weird today"* (continued sentence about Jarvis, not addressed to Jarvis).

3. **Media-playing threshold elevation.** If the system detects sustained background audio (TV, podcast, music — characterized by audio-energy fingerprint), it raises the wake-word confidence threshold by 30%. Catches Iron Man references on TV and podcast hosts named Jarvis.

4. **Configurable wake-word detector.** Use openWakeWord or Porcupine with custom-trained models for each name. Custom training reduces false-wake rate substantially over default models, particularly for "Jarvis" (a real name that appears in lots of media).

**No hardware kill-switch in v4.** The four self-suppression layers are sufficient. The schema for a hardware kill-switch is included as commented-out config for retrofit:

```yaml
# Optional: hardware kill-switch (mic mute via wall-mounted ESP32 button).
# Uncomment and configure if false-wake rate is unacceptably high in practice.
# kill_switch:
#   esp32_node: living_room_node_1
#   gpio_pin: 23
#   mute_duration_seconds: 300
```

### 30.4 Routing

```python
# modules/voice/wake_router.py

class WakeRouter:
    def __init__(self, persona_registry, orchestrator):
        self.personas = persona_registry
        self.orchestrator = orchestrator

    async def on_wake(self, word: str, audio_buffer: bytes):
        wake_config = self._lookup_wake_config(word)
        persona = self.personas.get(wake_config.persona)

        # Run the addressivity check.
        if not await self._is_addressed(audio_buffer):
            return  # silent abort

        # Route to orchestrator with persona context.
        await self.orchestrator.handle_query(
            audio=audio_buffer,
            persona=persona,
            wake_word=word,
        )
```

The orchestrator already exists; the persona is just an additional context parameter that gets composed into the prompt and selects the TTS voice for the response.

### 30.5 Alarm announcements are not persona-bound

Critical: the alarm subsystem (§29) speaks with its own *fixed* voice, not any persona's voice. *"BACK DOOR. SNEAKY IS OUTSIDE."* is system-level audio, not Jarvis or Mira or GLaDOS. Alarms are not conversation; they're hard safety-system output. Personas exist only for two-way interactive responses.
```

---

## EDIT 14 — §31 Notification Dispatcher (NEW TOP-LEVEL SECTION)

**Anchor:** insert after §30.

**INSERT:**

```markdown
## 31. Notification Dispatcher

Unified phone-alert dispatcher with three channels wired in v4: **ntfy**, **Telegram**, and **Home Assistant**. All three are enabled at install time so Cole can test them and decide which to keep on. Each channel can be toggled per-alarm-type.

### 31.1 Architecture

```python
# modules/notifications/dispatcher.py

@dataclass
class Alert:
    alarm_type: str          # "fire" | "cat_escape" | "door_open"
    title: str               # "Cat Escape: Sneaky"
    body: str                # "Sneaky exited via back_door at 14:32."
    priority: str            # "urgent" | "high" | "normal"
    metadata: dict           # alarm-specific extras

class NotificationDispatcher:
    def __init__(self, config):
        self.channels = []
        if config.ntfy.enabled:
            self.channels.append(NtfyChannel(config.ntfy))
        if config.telegram.enabled:
            self.channels.append(TelegramChannel(config.telegram))
        if config.home_assistant.enabled:
            self.channels.append(HAChannel(config.home_assistant))
        self.routing = config.routing  # per-alarm-type per-channel toggles

    async def send(self, alert: Alert):
        targets = [c for c in self.channels
                   if self.routing.is_enabled(alert.alarm_type, c.name)]
        results = await asyncio.gather(
            *[c.send(alert) for c in targets],
            return_exceptions=True,
        )
        for c, r in zip(targets, results):
            await self._log_delivery(c.name, alert, r)
```

Channels run in parallel; one failure doesn't block others. Per-channel delivery success/failure is logged for the dashboard.

### 31.2 Channel implementations

**ntfy** (recommended primary, self-hostable):

```python
class NtfyChannel:
    name = "ntfy"

    async def send(self, alert: Alert):
        priority_map = {"urgent": "urgent", "high": "high", "normal": "default"}
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{self.config.server}/{self.config.topic}",
                data=alert.body,
                headers={
                    "Title": alert.title,
                    "Priority": priority_map[alert.priority],
                    "Tags": self._tags_for(alert.alarm_type),
                },
            )

    def _tags_for(self, alarm_type):
        return {
            "fire": "rotating_light,fire",
            "cat_escape": "rotating_light,cat",
            "door_open": "door,warning",
        }[alarm_type]
```

ntfy server runs in a Docker container on the same box as Jarvis. Free, self-hosted, no external dependency. Phone app subscribes to a secret-ish topic name. **Urgent priority overrides DND** on Android reliably; iOS works for critical alerts but is slightly more restrictive.

**Telegram** (good for log/history channel, less reliable for DND override):

```python
class TelegramChannel:
    name = "telegram"

    async def send(self, alert: Alert):
        token = os.environ[self.config.bot_token_env]
        chat_id = os.environ[self.config.alert_chat_id_env]
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": f"*{alert.title}*\n{alert.body}",
                    "parse_mode": "Markdown",
                    "disable_notification": alert.priority == "normal",
                },
            )
```

Reuses the existing Mira-Telegram bot infrastructure (token already configured for that project; just add a separate chat for alerts).

**Home Assistant**:

```python
class HAChannel:
    name = "home_assistant"

    async def send(self, alert: Alert):
        url = f"{self.config.base_url}/api/services/notify/{self.config.service}"
        headers = {"Authorization": f"Bearer {os.environ[self.config.token_env]}"}
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, headers=headers, json={
                "title": alert.title,
                "message": alert.body,
                "data": {"priority": alert.priority, **alert.metadata},
            })
```

Requires an HA instance on the network. Worth running anyway for speaker integration, but the dependency cost is real if HA isn't otherwise on the roadmap.

### 31.3 Configuration

```yaml
notifications:
  ntfy:
    enabled: true
    server: http://localhost:8080      # self-hosted Docker
    topic: jarvis_household_alerts_<random_suffix>
  telegram:
    enabled: true
    bot_token_env: TELEGRAM_BOT_TOKEN
    alert_chat_id_env: TELEGRAM_ALERT_CHAT_ID
  home_assistant:
    enabled: true
    base_url: http://homeassistant.local:8123
    service: mobile_app_cole_phone
    token_env: HOME_ASSISTANT_TOKEN

  routing:
    fire:        [ntfy, telegram, home_assistant]
    cat_escape:  [ntfy, telegram, home_assistant]
    door_open:   [ntfy, home_assistant]   # skip telegram for lower-urgency
```

### 31.4 Dashboard test workflow

The notifications dashboard page surfaces:

- Per-channel enable toggle.
- Per-alarm-type per-channel routing matrix.
- Last 50 deliveries with timestamp, channel, alarm type, success/failure.
- **Test buttons**: *"Send test: cat escape via ntfy,"* *"Send test: fire via all,"* etc. Generates synthetic Alert payloads to compare arrival speed/visibility on Cole's phone.

Recommended workflow on first install: enable all three, fire each test alert, see which arrive cleanly on phone, decide which to disable for the alarm types where they're redundant.
```

---

## EDIT 15 — §28 Tunables additions

**Anchor:** §28, after existing subsections.

**INSERT** new subsections:

```markdown
### 28.5a Animal tracking (Section 22)

| Param | Default | Range | When to change |
|---|---|---|---|
| `tracked_species` | `[cat, dog]` | subset of `{cat, dog, ...}` | Add when a species joins the household. Removing disables both resident and visitor tracking for it. |
| `visiting_animal_retention_minutes` | 60 | 0 – 1440 | How long a transient stays after DEPARTED before reaping. |
| `auto_discovered_animal_naming` | `species_handle` | `species_handle | numbered | always_unknown` | How transient animals are named. |
| `dog_home_room_prior_weight` | 0.7 | 0.3 – 1.0 | Lower than cats — dogs follow humans more. |
| `cat_dynamic_size_window` | 200 | 50 – 1000 | Number of confident-attribution observations in the rolling-median size estimate. Higher = more stable, slower to adapt. |
| `cat_dynamic_size_cold_start_days` | 7 | 3 – 30 | Until this many days of history accumulate, the cost function uses static `expected_size` from config. |

### 28.5b Cat-escape alarm (Section 29.2)

| Param | Default | Range | When to change |
|---|---|---|---|
| `cat_escape_alarm.armed` | true | true / false | Don't change in config; disarm via runtime UX. Boot fails safe to armed. |
| `cat_escape_alarm.intentional_retrieval_window_seconds` | 30 | 0 – 300 | Window after alarm during which a human exit through the same door = retrieval, not unrelated. |
| `cat_escape_alarm.disarm_default_minutes` | 15 | 1 – 1440 | Default disarm duration on button-press; voice specifies own duration. |
| `cat_escape_alarm.suppress_default_minutes` | 30 | 1 – 240 | Default per-cat suppression duration for planned outings. |
| `cat_escape_alarm.mute_cooldown_minutes` | 5 | 1 – 60 | After voice-silence post-fire, this long before re-fire allowed. |
| `cat_escape_alarm.pre_arm_check_time` | "12:00" | "HH:MM" | Daily pre-arm check time (Cole's preference: noon). |
| `cat_escape_alarm.pre_arm_chime_volume` | 0.3 | 0.0 – 1.0 | Verification chime volume. |
| `cat_escape_alarm.fire_on_unidentified_cat` | true | true / false | Default true — better wrong-name than no-fire. |
| `cat_escape_alarm.override_quiet_hours` | true | true / false | Wakes you at 3 AM. Don't change. |

### 28.5c Door-open alarm (Section 29.3)

| Param | Default | Range | When to change |
|---|---|---|---|
| `door_open.unsupervised_grace_seconds` | 15 | 5 – 120 | Time door can be open with no human within radius before alarming. |
| `door_open.supervision_radius_m` | 3.0 | 1.0 – 10.0 | Radius around door door considered "supervising." Human within = no alarm. |
| `door_open.return_window_seconds` | 60 | 10 – 600 | After human exits via door, how long they have to come back before door alarm engages on the still-open door. |
| `door_open.t_off_frame_grace_seconds` | 8 | 2 – 30 | §13.x — how long after a human goes off-frame before they're considered absent. |

### 28.5d Fire alarm (Section 29.4)

| Param | Default | Range | When to change |
|---|---|---|---|
| `fire_alarm.signal_clearance_seconds` | 60 | 10 – 600 | Continuous-no-detection window required for natural resolve. |
| `fire_alarm.visual_confirmation_dwell_seconds` | 3 | 1 – 30 | How long a human must remain in the fire room before "visual confirmation" can silence audio. |
| `fire_alarm.unattended_rearm_seconds` | 300 | 0 – 3600 | After silence, if condition still active and no human in room, re-fire. 0 disables. |
| `fire_alarm.signal_increase_override_threshold` | 1.3 | 1.1 – 3.0 | Multiplier on detection signal that forces re-fire even from MUTED. 1.3 = "30% worse than at silence time." |
| `fire_alarm.no_resilience_silence_seconds` | 30 | 10 – 120 | After signal-increase override re-fires, this long before any silence command works again. |
| `fire_alarm.broadcast_on_silence` | true | true / false | Whether to phone-alert other residents when audio is silenced. Don't disable. |

### 28.5e Wake words (Section 30)

| Param | Default | Range | When to change |
|---|---|---|---|
| `wake.t_post_speech_suppression_seconds` | 2 | 0.5 – 10 | Suppression window after system stops speaking. |
| `wake.t_addressivity_window_seconds` | 1 | 0.3 – 3 | How long after wake-word detection to verify the audio is addressed-to-system. |
| `wake.media_threshold_elevation_pct` | 30 | 0 – 100 | Confidence threshold elevation when sustained media audio detected. |

### 28.5f Notifications (Section 31)

| Param | Default | Range | When to change |
|---|---|---|---|
| `notifications.delivery_timeout_seconds` | 5 | 1 – 30 | Per-channel HTTP timeout. |
| `notifications.delivery_log_retention_days` | 30 | 7 – 365 | How long delivery records stay queryable in the dashboard. |
```

---

## EDIT 16 — §32 Schema migrations summary (NEW)

**Anchor:** insert at end of doc.

**INSERT:**

```markdown
## 32. v4 Schema Migrations Summary

All v4 schema changes consolidated. Apply in order on first v4 boot.

```sql
-- migrations/20260512_v4_world_entities.sql

-- Already covered in §22.0/§25 if applied previously; safe to re-run.
ALTER TABLE world_entities ADD COLUMN IF NOT EXISTS is_resident BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE world_entities ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP NULL;

-- v4-specific.
ALTER TABLE world_entities ADD COLUMN IF NOT EXISTS household_owner_id INTEGER REFERENCES persons(id) ON DELETE SET NULL;
ALTER TABLE world_entities ADD COLUMN IF NOT EXISTS unmonitored_home_room TEXT NULL;

CREATE TABLE IF NOT EXISTS pet_affinities (
  pet_entity_id   INTEGER NOT NULL REFERENCES world_entities(id) ON DELETE CASCADE,
  person_id       INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  strength        TEXT NOT NULL CHECK (strength IN ('low','medium','high')),
  contexts        TEXT NOT NULL,
  PRIMARY KEY (pet_entity_id, person_id)
);
CREATE INDEX IF NOT EXISTS idx_pet_affinities_pet ON pet_affinities(pet_entity_id);

-- migrations/20260512_v4_alarms.sql

CREATE TABLE IF NOT EXISTS alarm_state (
  alarm_type      TEXT PRIMARY KEY,
  state           TEXT NOT NULL,
  state_since     TIMESTAMP NOT NULL,
  metadata        JSON
);

CREATE TABLE IF NOT EXISTS alarm_fires (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  alarm_type      TEXT NOT NULL,
  fired_at        TIMESTAMP NOT NULL,
  resolved_at     TIMESTAMP NULL,
  resolution      TEXT NULL,    -- 'condition_clear' | 'voice_silence' | 'visual_confirm' | 'manual'
  metadata        JSON
);
CREATE INDEX IF NOT EXISTS idx_alarm_fires_type_time ON alarm_fires(alarm_type, fired_at DESC);

CREATE TABLE IF NOT EXISTS notification_deliveries (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  alarm_fire_id   INTEGER NOT NULL REFERENCES alarm_fires(id) ON DELETE CASCADE,
  channel         TEXT NOT NULL,
  delivered_at    TIMESTAMP NOT NULL,
  success         BOOLEAN NOT NULL,
  error           TEXT NULL
);
CREATE INDEX IF NOT EXISTS idx_notification_deliveries_fire ON notification_deliveries(alarm_fire_id);

-- migrations/20260512_v4_door_state.sql

CREATE TABLE IF NOT EXISTS door_state (
  door_id         TEXT PRIMARY KEY,
  state           TEXT NOT NULL CHECK (state IN ('open','closed','unknown')),
  state_since     TIMESTAMP NOT NULL,
  source          TEXT NOT NULL CHECK (source IN ('vision','reed','manual'))
);
```

Schema validation on boot enforces:

- `pets.cats[*].household_owner` and `pets.dogs[*].household_owner` reference valid `residents[*].id`.
- Affinity contexts are in the enum: `sleeping | physical_contact | rubbing | proximity_general | authority | feeding`.
- Every `exterior_exit` polygon has a `display_name` field (used in alarm announcements).
- An `exterior_exit` polygon cannot share coordinates with an `outdoor_zone` polygon (the dog/cat outdoor semantic distinction from §22.10).
- The `tracked_species` list and `pets.<species>:` blocks are consistent (declaring dogs without `dog ∈ tracked_species` is a config error).
- Persona `system_prompt_file` paths exist and are readable.
- Wake words are unique across the `wake_words:` list.
- Notification channel `bot_token_env` / `token_env` environment variables are set if the channel is enabled.
```

---

That's v4. Apply all sixteen EDITs against the original `new_2.md` and the doc is current. v1, v2, and v3 patches are obsolete.
