SOPS = """
1. Write concisely, one idea per sentence.
2. Present facts first, add perspective later.
3. Use concise, single-topic questions for headings.
4. Group broad topics under H2, specific questions under H3.
5. Explain visual data before introducing tables.
6. Use active sentences: "X does Y" not "X is known for Y."
7. Back up claims with studies or data where possible.
8. Use precise wording for factual statements.
9. Avoid ambiguous or overly complex language.
10. Use specific numbers instead of vague quantities.
11. Always provide examples after plural nouns.
12. Simplify language, eliminate unnecessary complex words.
13. Bold the main answer and start with a direct response.
14. Use consistent structure in lists; use bullet points for readability.
15. Start instructions directly without process descriptions.
16. Clarify verbs involving multiple parties.
17. Use 'you' instead of 'I' throughout.
18. Use specific, action-oriented language for instructions.
19. Echo specific adjectives from questions in answers.
20. Start yes/no answers with yes or no.
21. Avoid linking sentences with 'ing' verbs.
22. End list introductions with a period, not a colon.
23. Include internal links to related articles for topical clustering.
24. Incorporate LSI keywords and related terms throughout.
25. Use visuals and structured data where applicable; introduce them with a brief description.
26. Optimize for featured snippets: direct answer within first 40-50 words of a section.
27. Use comparative analysis with tables or bullet points where relevant.
28. Include clear CTAs at the conclusion and contextually appropriate body sections.
29. Use synonyms and keyword variations for semantic coverage.
30. Use the PAS formula for introductions (Problem, Agitate, Solution).
31. Make content easy to skim: 2-4 sentence paragraphs, bullet points, bold key answers.
32. Use bucket brigades to keep content conversational (e.g. "Let me explain:", "The best part?").
33. Add at least one expert quote for credibility.
"""

AI_VISIBILITY_GUIDE = """
1. Entity Association: Define entities clearly with schema @id and sameAs to Wikidata/DBpedia. Structure around entity-attribute relationships, not just keywords.
2. Semantic HTML: Use descriptive nested headings (H1>H2>H3), well-formatted lists and tables, definition blocks after headings. Include a table of contents for long articles.
3. Information Gain: Include original data, unique statistics, or insights not found on other top pages. The AI cites the source that first introduced a unique fact.
4. Factual Consistency: Cite authoritative external references. Date content and attribute authors with credentials using schema markup.
5. Schema Markup: Use FAQ, HowTo, QAPage for question formats. Use Dataset/Table for data pages. Use Speakable to mark key paragraphs. Markup must match visible content exactly.
6. Contextual Relevance: Use natural language Q&A formats. Cover related subtopics an AI might need. Use semantic triplets (subject-predicate-object) naturally in text.
7. Freshness: Add machine-readable last-updated timestamps. Update data points and year references regularly. Show content is actively maintained.
8. Publisher Authority: Consistent brand signals across the web. Transparent authorship. Implied links and brand mentions in authoritative sources matter beyond backlinks.
9. Technical: Fast server response, content in initial HTML (not JS-only), correct robots.txt with Google-Extended token.
10. Engagement: Answer immediately (inverted pyramid). Reduce pogo-sticking with direct answers up front. Format for low abandonment.
"""
