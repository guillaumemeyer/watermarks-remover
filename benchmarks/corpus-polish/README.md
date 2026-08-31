# Polish benchmark corpus

## Purpose

This corpus contains short, self-contained texts written originally in Polish. It is intended for rewrite evaluation in a language whose grammar relies heavily on word endings.

A reader does not need to know Polish grammar to use the corpus. In simple terms, Polish changes the endings of nouns, adjectives, pronouns and verbs to show who does something, who receives it, what is affected, where something happens, and whether an action is ongoing or completed. A rewrite may preserve the general meaning while damaging these relationships. The metadata below explains which relationship is especially visible in each text.

The texts are not translations of the English corpus. The first eight texts cover comparable subject areas where practical, but they were written independently in natural Polish.

## Released material

The release contains three clearly separated groups:

- corpus-polish/: 50 main texts used in the standard Polish run.
- excluded-numeric/: 4 controlled texts containing digits, marked EXCLUDED-NUMERIC.
- additional/: 3 optional cultural or register samples, marked ADDITIONAL.

Replaced drafts and reserve texts are not part of the repository release. They remain in the author's working archive and are not listed in this README.

## File format

Each .txt file contains only one Polish paragraph. It has no title, identifier, metadata, frontmatter, instruction or comment inside the file.

Main-corpus filenames begin with a two-digit number, from 01_chmura-obliczeniowa.txt to 50_dziecko-i-dziewczynka.txt. The numeric prefix fixes the order after alphabetical sorting.

Files must be UTF-8, non-empty after trimming, and smaller than 64 KB. Every released text contains 50–90 Polish words and ends as a complete paragraph. Metadata belongs only in this README, never in the benchmark input files.

## Directory layout

    benchmarks/
    └── corpus-polish/
        ├── README.md
        ├── 01_chmura-obliczeniowa.txt
        ├── ...
        ├── 50_dziecko-i-dziewczynka.txt
        ├── excluded-numeric/
        │   ├── X01_zaginione-dziecko-cyfry.txt
        │   ├── X02_zwierzeta-w-schronisku-cyfry.txt
        │   ├── X03_dostawa-do-ksiegarni-cyfry.txt
        │   └── X04_grupa-wycieczkowa-cyfry.txt
        └── additional/
            ├── A01_gorzka-herbata-w-kubku.txt
            ├── A02_zmiana-organizacji-ruchu.txt
            └── A03_kontrola-instalacji.txt

The exact repository path may be adjusted to match the benchmark loader. The separation between the three groups must be preserved.

## How to read the grammar labels

The labels describe ordinary Polish behavior rather than instructions inserted into the texts.

| Label | Plain-English meaning |
|---|---|
| GEN | A noun ending often marks absence, possession, quantity, source or a relation between nouns. |
| DAT | A noun or pronoun ending marks the receiver or beneficiary of an action. |
| ACC | A noun or pronoun ending marks the person or thing directly affected by an action. |
| INS | A noun ending marks a tool, means of transport, companion, collaborator or role. |
| LOC | A noun ending is used after selected prepositions for locations and topics of speech or thought. |
| ADJ-N | An adjective changes to match its noun in gender, number and sentence role. |
| VERB | A verb changes for person, number, tense and sometimes gender. |
| PRON | A pronoun changes form when the same person takes a different role in the sentence. |
| MIXED | Several linked changes occur around stable people or objects, making relationship errors easier to observe. |
| PARITY | The topic corresponds broadly to an English-corpus domain, but the Polish text was written independently. |
| aspect pair | Polish commonly distinguishes an ongoing, repeated or unfinished action from a completed action, often with two related verb forms. |
| numeral agreement | The form of a noun and nearby words can change according to the quantity. |

These labels identify the most useful feature of a text. They do not claim that the text contains only that feature. Nominative is not a separate target because it is the ordinary subject form and occurs throughout Polish text. Vocative is not a separate target because it is limited mainly to direct address and would require an unusually narrow set of situations.

## Why digits are separated

Polish uses both digits and number words in normal writing. The grammatical form of the following noun can depend on the number, including the final digit of a compound number. A blanket ban on digits would therefore remove a real source of variation.

The four digit-bearing texts are released separately so their effect can be measured and reported without changing the default corpus. They are included in the repository but excluded from the main 50-text run.

## Corpus design

The corpus favors natural distribution over grammatical saturation. No text was written to contain every phenomenon at once. Related features are spread across the set.

Current main-corpus coverage:

- verb aspect pairs: 13 of 50 texts (26%);
- numeral–noun constructions: 15 of 50 texts (30%);
- multiple noun cases and agreement patterns: distributed across the corpus.

The texts use contemporary Polish and mostly everyday, informative language. Two optional samples deliberately use bureaucratic Polish to test whether a rewrite preserves formal register. One optional sample replaces the English coffee-brewing concept with a culturally familiar Polish tea-preparation scene.

## Main corpus metadata

Status: MAIN. These 50 files form the default Polish corpus.

| File | Category | Main target | What this exposes | Related set | Words |
|---|---|---|---|---|---:|
| 01_chmura-obliczeniowa.txt | PARITY | mixed noun cases; verb aspect | Noun endings mark relations between a seasonal business, its files, workers and service provider. | — | 59 |
| 02_parzenie-kawy.txt | PARITY | adjective–noun agreement; instrumental | Adjectives follow several nouns; instrumental forms name tools and materials used during preparation. | — | 63 |
| 03_przygotowanie-do-wedrowki.txt | PARITY | accusative; instrumental; verb aspect | Objects name equipment and waste; instrumental forms mark companionship and means of safe movement. | — | 58 |
| 04_mity-zywieniowe.txt | PARITY | genitive; adjective–noun agreement | Genitive forms express causes, quantities and lack of evidence; adjectives follow nouns in several roles. | — | 62 |
| 05_licencje-otwartego-oprogramowania.txt | PARITY | industry compounds; aspect pair: sprawdzać–sprawdzić | Technical compounds occur with noun forms marking authorship, dependency, modification and redistribution. | — | 54 |
| 06_energia-odnawialna.txt | PARITY | industry compounds; verb aspect | Technical compounds occur naturally; verb aspect distinguishes repeated forecasting from completed balancing actions. | — | 56 |
| 07_finanse-malej-firmy.txt | PARITY | genitive; adjective–noun agreement | Noun and adjective endings mark invoices, reserves, stock and several kinds of business expense. | — | 55 |
| 08_historia-wenecji.txt | PARITY | locative; past-tense gender and number | Locative forms describe canals, squares and buildings; past-tense verbs agree with the city and its residents. | — | 58 |
| 09_brak-skladnika-w-piekarni.txt | GEN | genitive after absence and negation | Polish changes the noun ending after expressions of absence and after many negated verbs. | — | 57 |
| 10_naprawa-starej-kuchni.txt | GEN | genitive relations; aspect pair: naprawiać–naprawić | Genitive endings link parts of the kitchen and household objects; the same repair verb appears as an ongoing and completed action. | — | 53 |
| 11_zimowe-zapasy.txt | GEN | quantity and genitive plural | Quantity expressions cause characteristic noun forms, especially when the amount is large or not precisely stated. | — | 51 |
| 12_podroz-bez-bagazu.txt | GEN | prepositions requiring genitive | Several Polish prepositions require a changed noun ending, especially equivalents of without, from, to and during. | — | 56 |
| 13_opieka-nad-ogrodem.txt | GEN | need, avoidance and nominal relations | Genitive forms appear after need, avoidance and noun-to-noun relations. | — | 56 |
| 14_pomoc-w-schronisku.txt | DAT | recipient and beneficiary | Dative forms mark the person or animal receiving food, help, information or another benefit. | animal-care-01 | 63 |
| 15_nauka-pracy-w-sklepie.txt | DAT | recipient; aspect pair: tłumaczyć–wytłumaczyć | Dative forms identify people receiving explanations and help; one explanation continues and is then completed. | — | 51 |
| 16_przygotowanie-szkolnego-festynu.txt | DAT | assigning to recipients; aspect pair: rozdawać–rozdać | Dative forms show who receives tasks and materials; the same distribution verb marks a process and its completion. | — | 53 |
| 17_awaria-w-pensjonacie.txt | DAT | answering and communicating to recipients | The endings distinguish guests receiving explanations, alternatives and reassurance. | — | 57 |
| 18_rodzinny-posilek.txt | DAT | beneficiary and interpersonal receiver | Different family members receive meals, alternatives, advice and explanations. | — | 54 |
| 19_pakowanie-przed-przeprowadzka.txt | ACC | direct objects; aspect pair: pakować–spakować | Accusative forms mark objects handled during a move; packing appears first as a process and then as a completed task. | — | 51 |
| 20_zakup-roweru.txt | ACC | direct object: animate and inanimate | The same case marks a selected object and a person directly affected by an action. | — | 52 |
| 21_badanie-psa.txt | ACC | direct target of care | Accusative forms mark the animal and objects directly examined, prepared, moved or treated. | animal-care-01 | 55 |
| 22_planowanie-podrozy.txt | ACC | planned route and destination | Accusative endings mark the route, destination, supplies and actions chosen for a future journey. | — | 53 |
| 23_poszukiwanie-turysty.txt | ACC | animate direct object; aspect pair: przeszukiwać–przeszukać | Accusative forms identify the missing person and searched area; the same search verb marks an ongoing and completed action. | — | 51 |
| 24_spacer-z-psem.txt | INS | companion, tool and co-worker | Instrumental forms mark who accompanies the worker, which tool she uses and whom she cooperates with. | animal-care-01 | 52 |
| 25_czyszczenie-starego-stolu.txt | INS | tools and means; aspect pair: czyścić–wyczyścić | Instrumental endings name tools and materials; cleaning appears as both an ongoing and completed action. | — | 54 |
| 26_podroz-na-wyspe.txt | INS | means of transport | Polish uses instrumental forms for travelling by train, bus, bicycle, ferry or another means. | — | 50 |
| 27_wspolna-wystawa.txt | INS | cooperation and roles; aspect pair: przygotowywać–przygotować | Instrumental forms mark collaborators and professional roles; preparation is shown during the work and after completion. | — | 54 |
| 28_wyjazd-grupy-nad-jezioro.txt | INS | means of transport; compound numeral ending in two–four | A compound number written as words combines with a plural noun while instrumental forms name several means of transport. | — | 55 |
| 29_rozmowa-o-przeprowadzce.txt | LOC | topic of conversation and thought | Locative forms appear after selected prepositions when people talk or think about something. | — | 55 |
| 30_odnowienie-oranzerii.txt | LOC | locations; aspect pair: odnawiać–odnowić | Locative forms place work in several parts of a building; renovation appears as a process and a completed result. | — | 51 |
| 31_rozmowa-o-osiedlu.txt | LOC | discussion topics; aspect pair: omawiać–omówić | Locative forms introduce several everyday topics; discussion appears while continuing and after reaching a conclusion. | — | 56 |
| 32_wspomnienie-o-wycieczce.txt | LOC | events and people as topics | Locative forms let the narrator speak about former pupils, visited places and remembered events. | — | 52 |
| 33_mieszkania-w-starej-kamienicy.txt | LOC | plural locations; compound numeral ending in two–four | The compound number is written in words and combines with a plural noun; locative endings describe life in shared spaces. | — | 57 |
| 34_wystawcy-na-kiermaszu.txt | ADJ-N | adjective agreement; compound numeral ending in five–nine | A compound number written as words requires a different noun form; adjective phrases vary across gender and number. | — | 56 |
| 35_rodzenstwo-w-pracowni.txt | ADJ-N | agreement describing people | Adjectives agree with a feminine person, a masculine person and plural groups in different roles. | — | 55 |
| 36_siedem-domkow-nad-jeziorem.txt | ADJ-N | agreement after quantity above four | A number greater than four combines with a plural noun and several adjective phrases in everyday description. | — | 55 |
| 37_urzadzanie-nowego-mieszkania.txt | ADJ-N | adjective agreement; aspect pair: urządzać–urządzić | Adjectives follow nouns of different genders and numbers; furnishing appears as an ongoing and completed action. | — | 55 |
| 38_zimowy-poranek-w-schronisku.txt | ADJ-N | dense natural agreement | Multiple adjective–noun phrases cover gender, number and several cases without explicit grammar instructions. | — | 62 |
| 39_poranna-rutyna.txt | VERB | present tense; different persons | Verb endings change with first, second and third person in singular and plural. | — | 55 |
| 40_wczorajsza-wyprawa.txt | VERB | past tense; gender and number | Past-tense verb endings reveal whether the subject was a woman, man or group. | — | 55 |
| 41_przyszly-ogrod.txt | VERB | future tense; singular and plural | Future forms show what one person and several people will do, start doing or finish. | — | 52 |
| 42_rozmowa-po-seansie.txt | VERB | aspect pair: oglądać–obejrzeć; past tense gender and number | The same viewing verb marks an ongoing and completed action; past-tense forms distinguish a woman, a man and the pair. | — | 58 |
| 43_wybor-filmu-i-biletow.txt | VERB | aspect pair: wybierać–wybrać; preferences and completed decision | The same choice verb marks deliberation and its result; two people express preferences before buying tickets. | — | 60 |
| 44_trudna-decyzja.txt | PRON | first-person pronouns | The same speaker appears as subject, receiver, object and companion, so the pronoun repeatedly changes form. | — | 58 |
| 45_pierwszy-dzien-w-pracy.txt | PRON | second-person pronouns | The listener is addressed as subject, receiver, object and companion. | — | 55 |
| 46_zagubiona-pamiatka.txt | PRON | third-person masculine pronouns | A male referent remains the same while pronoun forms change with his role in each sentence. | — | 55 |
| 47_siostry-w-pensjonacie.txt | PRON | third-person plural feminine reference | A group of women is referred to with plural pronouns in several syntactic roles. | — | 56 |
| 48_pierwsza-randka.txt | MIXED | stable referents; changing case, gender and pronouns | The boy and girl stay the same, but nouns, pronouns and verb endings change whenever their roles change. | date-01 | 57 |
| 49_zaproszenie-na-spotkanie.txt | MIXED | first person, gender and interpersonal roles | The speaker changes between subject, receiver, object and companion; masculine and feminine verb forms also appear. | date-01 | 62 |
| 50_dziecko-i-dziewczynka.txt | MIXED | same person described with neuter and feminine nouns | One child is first named with a neuter noun and then a feminine noun, forcing verbs and descriptions to change. | referent-gender-01 | 50 |

## Digit-bearing metadata

Status: EXCLUDED-NUMERIC. These files are released for a separate controlled run and must not be mixed silently into the main result.

| File | Category | Main target | What this exposes | Related set | Words |
|---|---|---|---|---|---:|
| X01_zaginione-dziecko-cyfry.txt | NUMERIC-EXCLUDED | digit–noun agreement; gender and name variation | The same child is described with a neuter noun, a feminine noun, a full name and a nickname. A digit expresses age. | — | 55 |
| X02_zwierzeta-w-schronisku-cyfry.txt | NUMERIC-EXCLUDED | digit–noun agreement after small and larger quantities | Polish uses different noun forms after quantities ending in two to four and after quantities of five or more. | — | 54 |
| X03_dostawa-do-ksiegarni-cyfry.txt | NUMERIC-EXCLUDED | compound numbers ending in two–four versus five–nine | The final digit of a compound number changes the noun form, so similar quantities create different grammar. | — | 52 |
| X04_grupa-wycieczkowa-cyfry.txt | NUMERIC-EXCLUDED | one, small quantities and larger quantities | The text contrasts singular agreement with forms used after two to four and after five or more. | — | 52 |

## Additional-sample metadata

Status: ADDITIONAL. These files are optional and must be reported separately from the main corpus.

| File | Category | Main target | What this exposes | Related set | Words |
|---|---|---|---|---|---:|
| A01_gorzka-herbata-w-kubku.txt | CULTURAL-PL | everyday Polish register; preparation sequence | A familiar Polish way of making tea creates ordinary noun and verb forms without copying the English coffee passage. | cultural-drink-01 | 57 |
| A02_zmiana-organizacji-ruchu.txt | REGISTER-FORMAL | administrative nominalisations and impersonal constructions | Deliberately bureaucratic Polish checks whether a rewrite preserves an official register instead of simplifying it. | formal-register-01 | 50 |
| A03_kontrola-instalacji.txt | REGISTER-FORMAL | administrative passive and impersonal syntax | Deliberately bureaucratic Polish contains passive wording, abstract nouns and extended dependencies. | formal-register-01 | 53 |

## Validation checklist

Before committing the corpus, verify that:

- the main directory contains exactly 50 numbered .txt texts plus this README;
- each benchmark text file contains only its Polish paragraph;
- every text has 50–90 Polish words;
- every paragraph is complete and self-contained;
- no main text contains a digit, URL or block quote;
- the four digit-bearing files remain in excluded-numeric/;
- the three optional files remain in additional/;
- reserve and replaced drafts are absent from the repository;
- metadata are present only in this README;
- filenames and folder membership match the tables above.

## Citation

Add the OSF preprint citation and permanent link here before the final pull request.

