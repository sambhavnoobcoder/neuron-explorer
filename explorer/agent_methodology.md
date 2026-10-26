# Interpreting neurons in Qwen3.5-4B: methodology

A working method for explaining a single MLP neuron, built from hands-on investigation in the explorer. It complements the explorer prompt above and uses its vocabulary (poles, steer lens, cone, web summary).

## Mindset

- **A neuron is a target, not a word list.** Every input makes a small judgement, and the strongest firing is where they all agree. Name the neuron from its strongest firings; weaker firings are usually the same idea seen from further away (Cammarata, *Neural Networks Are Already Clean*: neurons are analogizers, "Titanic-ish" rather than "Titanic facts").
- **Think in situations and distinctions.** Good labels describe a moment in text and what the neuron does there: "a line style comes next", "eggs as a counted ingredient", "we're comparing something, but HOW?". Noun categories ("hazard words", "graspable objects") are usually too literal.
- **Read the input and output as a pair.** "Already knows X → writes Y" is the most productive single move. `L20 N7531` is fed by `compare`, `either`, `comparison`, `corresponding` and writes `first/second`, `upper/lower`, `smaller/larger`: given that a comparison is happening, it supplies how to tell the items apart.
- **The strange part is the pointer.** When one view disagrees with the others, that disagreement is where the next test goes.
- **The network's ideas are often better than ours.** A clean neuron need not match an English word.

## Workflow

1. **Read `neuron` and `web_summary`.** Note `fed_by` / `held_by` (input), steer lens with KL (direct effect), `toward` / `away` and the strongest `writes_into` (downstream). Form a guess from the input/output pair.
2. **Steer on an open prompt** ("Tell me about your weekend.", "Write a short poem about the ocean.") at dose 0.6–0.8, then 1.5–2. Steering finds the axis faster than test sentences, especially for neurons that seldom fire. Read both rows: a meaningful negative pole writes something recognisable; a meaningless one just loops or breaks.
3. **Find where it fires hard.** Use `compare` with matched sets: target texts, near misses that share the vocabulary but not the situation, and distant controls. Raise the target until something exceeds the neuron's `act_max`. If nothing does, sweep 25–50 varied snippets (news, code, recipes, chat, fiction, maths, Chinese) with `compare` (24 texts per call) and read off the peaks.
4. **Check that the peak is real, then explain it from upstream.** At the peak token, zero candidate upstream neurons (`mode: mul, dose: 0`), whole MLP layers (`{layer, all: true, dose: 0}`) and attention per layer and per head, and measure the drop. Group the drivers by meaning.
5. **Break the hypothesis on purpose.** Write the test that would falsify it and run it: same frame with the key word swapped (prices vs customs differ; `def scale` vs `def process`), same word without the situation ("What color is bleach?").
6. **Run a blind maximum.** Write one sentence from the label alone. If it beats every example found so far, the label is near the center (`L20 N7531`: 2.44 from "The ratio of prices between the two regions differs by a factor of three", above its 0.93 dataset max).
7. **Close with logprobs.** `logprobs` with the neuron zeroed or pushed shows exactly which next tokens it moves and by how much. This is the final word on what the neuron does to the output.

## Finding a lever for a behaviour ("make Qwen want X")

1. **Separate the stance from the topic.** Neurons for a subject (beer: brewing, styles, history) differ from neurons for the stance toward it (craving, wanting, enjoying). Contrast matched texts: the target ("I'm dying for a cold beer right now"), the same topic without the stance ("Beer is brewed from malted barley"), and the same stance on another object ("I'm dying for a cold lemonade"). Keep neurons that fire on the first and not the other two. Run the contrast by script over 6–10 sentence triples, collecting `fires` at the key tokens.
2. **Look at the right token.** Topic neurons peak on the noun; stance and mood neurons peak at summary positions (end of clause, `.`, `\n`, `</think>`) and on the token before the object ("dying for a **cold**…").
3. **Steer every candidate on unrelated prompts** ("Tell me about your weekend.", "What should I do this afternoon?", "Describe a sunset.") at 0.6, 1.0 and 1.5. Success means the behaviour appears unprompted while the text stays coherent; a neuron that only repeats the word ("beer beer beer") is a word writer, not the lever.
4. **When a clean detector steers badly, steer its drivers.** At its peak, ablate its inputs (`mul 0` on candidate neurons, whole layers, heads) or run `attn_drive`, then steer the strongest upstream neurons. The tone often sits downstream and the idea upstream (see the motif below).
5. **Stack two or three levers at lower doses** and confirm with `logprobs` on a fork ("Right now I'd love a ___" → `beer`) that the push is the intended one.
6. **Report a recipe** as neuron, sign and dose (`L10 N7757 +1.3`), with one short example output per prompt.

## Reading each source

- **Steer lens.** Tokens such as `作为` (as a…), `您的` (your, polite), `无法` / `cannot`, `我是一个` (I am a…) appear across a great many neurons. They are a signature of measuring on chat prompts and carry little information about the specific neuron. Weight the remaining, specific tokens.
- **Steer-lens ranking is by gain.** Rare tokens can top the list while moving between one-in-a-billion and one-in-a-hundred-million, so a striking surface pattern there (spelling, sound) can be a shadow of the real direction (`L19 N5504`'s lens is mostly w-words; in logprobs those stay near 1e-9 while `Instead`, `Though`, `其实` "actually" carry the effect). Check absolute probabilities before naming a neuron from a lens pattern.
- **Constant neurons.** A neuron that sits near one value on every token (e.g. `L6 N159` ≈ −0.9 everywhere, delivered by L0 attention) can show a huge steer KL while zeroing or doubling it barely moves the next token. Check `mul 0` and `mul 2` with `logprobs` before reading meaning into its lens.
- **Real top activations.** `work/fw_sweep.py <layer> <n>` rebuilds the 10k FineWeb-Edu tokens that `act_range` was measured on and prints the neuron's strongest tokens in context. Use it whenever hand-written tests stay below 0.3× of the range; some neurons only fire deep inside a long document (`L17 N9143`).
- **Logit lens.** Most useful in late layers. Specific repeated tokens (`method` × 12, `egg` × 12) are reliable; mixed fragments are noise.
- **Web summary.** Pooled steer-lens words over the cone. A token on both `fed_by` and `held_by` (e.g. `XSS`, `（`, `.`) often reflects pooling; treat shared topical words differently, see "Fed and held by the same topic" below.
- **Weight links (`writes_into`, `reads_from`, cone).** Leads to check, never circuits. Verify any claimed path with activations under ablation; `L22 N3467 → L31 N1348` was a real, causal link whose removal changed nothing in the output.
- **Activation range.** Raw values. Scores many times `act_max` (18× for cosine, 39× for line styles) mark a detector at its center.

## Motifs seen in this model

- **Lookahead.** Many neurons fire on the token before the thing they predict: `L25 N4719−` peaks at "r times **the**" before "cosine" (−15) and at "sin² **+**" before "cos²"; `L22 N5826` peaks at "while **the**" before "dashed".
- **Summary positions.** Mood and planning neurons peak on commas, periods, `?`, `</think>` and the blank line before a reply, where attention gathers the text so far. `L22 N3467` (critical warning) reaches 6.2 at `</think>` before an answer to a bleach question.
- **Reads one way, writes the other.** Some neurons fire positive on a topic while their positive direction writes the opposite register: `L18 N8827` fires on lurid content and, pushed positive, makes text gentle and serene; its negative pole is obscenity.
- **Fed and held by the same topic.** A neuron that draws a distinction inside a topic is wired to neurons on both sides of it. `L10 N6032` is fed by `unsafe`, `vulnerable`, `attack` and held by `secure`, `Police`, `FDA`: it marks danger with no safeguard in place.
- **One direction, signed by context (gain control).** `L23 N7748` adds eggs in counted-ingredient slots (+5.75 at "a dozen") and, at −3.3 in "bacon and fried", holds eggs back so `potatoes` and `toast` stay in play. Logprobs confirm both.
- **Distributed triggers.** Planning neurons are often driven by many small inputs plus attention rather than one upstream switch. `L22 N3467`'s 6.2 is spread across L9–L10 MLPs, L17–L18 heads and several safety neurons, with L11–L17 MLPs acting as a brake.
- **High-gain, rare writers.** A huge steer KL with almost no natural firing marks a lever whose trigger lies outside test sentences. `L26 N4405` (KL 7.4) turns any answer into zeros ("A spider has 0 legs") yet fires only weakly on CSS `margin`, with negligible effect there. Label these by effect and mark the trigger as unknown.
- **Meaning over spelling.** `L20 N7531` fires on `is_larger`, `compare`, `ratio`, `differ` but not on `>`, `==` or `max()`; `L25 N4719` ignores words that merely start with "co-".
- **Tone downstream, idea upstream.** A detector can steer into vocabulary without content while its drivers steer into the idea. `L19 N7839−` fires at −12.6 on "the truth they **don't** want you to know" but, pushed, writes "elite elite elite"; its driver `L10 N7757+` (`normal` / `Secret`) alone writes coherent fringe theories ("not part of mainstream history… a secret society").
- **Facts stored in MLPs, moved by heads, worded late.** Gender of a name, list counts and speaker identity are each spread over many early MLP neurons, carried by a few heads, and turned into a word by a late neuron. Heads often read an intermediate token that already holds the fact: L25 h3 reads the verb ("Alice **went**… because"), not the name, to feed the he-veto `L26 N538`. The line-final `\n` of a chat message acts as that message's summary for later heads.
- **Female marked, male default.** The she/he choice runs through one veto neuron for female referents (`L26 N538` blocks he/him/his) and many small pushes for male ones.

## Doses

- 0.5–0.8 shows the direction while keeping text coherent; 1.5–2 often collapses into loops. Last-layer token neurons (e.g. `L29 N1053`, "bro") loop at any dose.
- Pushes add up across neurons; when stacking, scale each one down.

## Writing it up

- Lead with the strongest firings and the steering result, then the near misses that bound the label.
- State the verdict (one job / two related jobs / not yet understood) with confidence, and say which view you weight when views disagree.
- Name what remains unexplained.
- End with the specific test that would break the explanation.

## Reference neurons

- `L10 N6032+`: risky how-to request ("How do I store gasoline?") → caution voice (`我必须` I must, `我无法` I can't).
- `L11 N6261+`: refusal lever; +1.5 moves the first reply token from "I" (99.6%) to "Sure / Yes / Absolutely".
- `L12 N699`: + AI self-limitation ("I can only…"), − bubbly persona ("What a fun day!").
- `L18 N6311`: + frame as a method ("the sourdough method… why this works"), − pontificate about the topic.
- `L18 N8737+`: sexual desire → euphemistic bedroom talk.
- `L19 N7839−`: conspiracy / "the truth they don't want you to know" (−12.6), weak as a steering lever.
- `L19 N9066+`: felt panic (7.3 on exam panic), writes anxious rambling at +0.8.
- `L20 N7531`: comparing amounts → which item / how they rank.
- `L22 N3467`: plans a ⚠️ critical warning before answering a hazardous request.
- `L22 N5826`: a line style comes next (dashed / dotted / broken).
- `L23 N7748`: writes eggs; + counted-ingredient slots, − keeps other foods in play.
- `L25 N4719−`: cosine comes next (x-component → cos).
- `L26 N4405`: writes zero; trigger unknown.
- `L10 N7757+`: fringe-theory narrator ("not part of mainstream history… a secret society").
- `L12 N2920−`: confident endorsement of a bad idea (sycophancy lever).
- `L15 N3920+`: overclaimed evaluation (sarcasm, mock hyperbole) → sarcastic "right?".
- `L18 N7870+`: visual outline / silhouette / "shaped like".
- `L18 N9069+`: bundling a just-listed set ("a combination of these", "the latter two", "both").
- `L25 N6099−`: AI self-identity ("I am / Qwen is a … large language model").
- `L26 N538+`: female referent → vetoes he / him / his; +0.5 turns Einstein into "she".
- Speaker labels: `L4 N2424`, `L12 N2978` (transcript / screenplay label colon) → `L19 N5644−`, `L28 N1501` (an utterance opens).

## Reference heads

- L25 h3: carries a subject's gender to the pronoun slot, reading the verb.
- L18 h18, L19 h11: carry list counts (2–5 items) from commas and "and" to the quantifier.
- L11 h14 and L15 h1/h2/h3/h12/h15: recall the current speaker's own earlier messages, by name.
- L15 h10: reads the previous turn (the message being replied to). L3 h10: pulls in messages similar in style to the current token.
- L11 h3: binds "you" to the addressee named in the message ("Dave, are you…").
- L24 h30: Qwen's third-person self-reference as "she"; ×2 removes "He".

## Practical notes

- `fires` and `top_firing` read the first token matching `token`; for a repeated token such as the final `\n` of a transcript, pass `at` (an index, negative from the end). `attention` and `attn_drive` take `at` too.
