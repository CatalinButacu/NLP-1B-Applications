# NLP Assignment 4 - Presentation Materials

## Part 3: Error Analysis of Fact-Checking Model

### Error Categories

1. **Semantic Confusion**: Model fails to distinguish between related but distinct concepts (e.g., engraver vs. printmaker)

2. **Implicit Information**: Model incorrectly infers information not explicitly stated in the passage

3. **Temporal Mismatch**: Model fails to correctly match temporal information between fact and passage

4. **Reference Resolution Failure**: Model fails to properly resolve references across different parts of the passage

### Aggregate Statistics

| Error Type | Category | Count |
|------------|----------|-------|
| False Positive | Semantic Confusion | 4 |
| False Positive | Implicit Information | 6 |
| False Negative | Temporal Mismatch | 5 |
| False Negative | Reference Resolution Failure | 5 |

### Key Examples

#### Example 1: Semantic Confusion (False Positive)

**Fact**: Jean Daullé was a printmaker.

**Passage**: "Jean Daullé (18 May 1703 – 23 April 1763) was a French engraver. [...] He engraved several portraits and plates of historical and other subjects..."

**Ground Truth**: Not Supported (NS)

**Model Prediction**: Supported (S)

**Error Explanation**: Model failed to distinguish between an engraver and a printmaker. While related professions involving similar skills, they are distinct occupations.

#### Example 2: Implicit Information (False Positive)

**Fact**: She has worked with a number of organizations.

**Passage**: "Hibo Wardere is a Somali-born campaigner against female genital mutilation (FGM), author, and public speaker. [...] She currently resides in Walthamstow, London, where she worked as a mediator and a regular FGM educator for Waltham Forest Borough."

**Ground Truth**: Not Supported (NS)

**Model Prediction**: Supported (S)

**Error Explanation**: Model incorrectly inferred that working as "a mediator and FGM educator for Waltham Forest Borough" means she worked with multiple organizations. The passage only explicitly mentions one organization.

#### Example 3: Reference Resolution Failure (False Negative)

**Fact**: With CSKA, he won domestic titles.

**Passage**: "Honours.:Club. Internacional - Campeonato Gaúcho: 2002, 2003 - Copa Sudamericana: 2008 CSKA Moscow - Russian Premier League: 2005, 2006 - Russian Cup: 2004–05, 2005–06, 2008–09 - Russian Super Cup: 2004, 2006, 2007, 2009 - UEFA Cup: 2004–05 Palmeiras - Copa do Brasil: 2012 Botafogo - Campeonato Brasileiro Série B: 2015 Goiás - Campeonato Goiano: 2016"

**Ground Truth**: Supported (S)

**Model Prediction**: Not Supported (NS)

**Error Explanation**: Model failed to recognize that the listed achievements under "CSKA Moscow" (Russian Premier League, Russian Cup, Russian Super Cup) are domestic titles.

## Part 4: LLM Output Analysis

### Prompt 1: Research Proposal on Social Media and Mental Health

**Prompt**: Create a comprehensive research proposal for a study on the impact of social media on mental health among college students. Include literature review, methodology, and implications.

#### Comparative Analysis

| Aspect | Model 1 | Model 2 |
|--------|---------|--------|
| **Comprehensiveness** | Exceptional depth in all sections, particularly methodology with detailed data collection methods and analysis approaches | Covers essential elements but lacks depth in methodology and sample size justification |
| **Structure** | Formal academic structure with clear hierarchical organization | Logical structure but lacks formal academic formatting with numbered sections |
| **Practical Application** | Good coverage of implications but less specific | Good connection to practical applications with examples for educational interventions, clinical applications, and policy development |

### Prompt 2: Conference Planning Guide

**Prompt**: Write a detailed guide for organizing a three-day international conference on sustainable urban development, including planning timelines, budget considerations, speaker selection, and sustainability strategies.

#### Comparative Analysis

| Aspect | Model 1 | Model 2 |
|--------|---------|--------|
| **Organization** | Strong structure with detailed sections and subsections | Good organization but less comprehensive in certain areas |
| **Practicality** | Realistic 12-18 month planning timeline with detailed milestones | Compressed 6-month timeline that seems less realistic for international event planning |
| **Sustainability Focus** | Comprehensive sustainability measures integrated throughout all aspects | Good sustainability coverage but less integrated into overall planning |

### Identified Flaws in Model 2's Responses

1. **Research Proposal**: Lacks critical ethical and technical details regarding social media usage tracking. The proposal mentions installing a tracking application but doesn't adequately address privacy concerns, data security measures, or technical limitations of tracking across multiple platforms.

2. **Conference Planning Guide**: Presents a compressed planning timeline (6 months) for an international conference. Industry standards suggest 12-18 months for events of this scale to secure venues, international speakers, and sponsors.