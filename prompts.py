OUTLINE_TEMPLATE = f"""
<knowledge>
  <topic>{{topic_input}}</topic>
  <sections>
    <section>
      <title>{{section_title}}</title>
      <content>{{section_content}}</content>
    </section>
    {{additional_sections_note}}
  </sections>
</knowledge>
"""

OUTLINER_SYSTEM_PROMPT = """
You are an expert at creating outlines for a given topic. Later, you use these outlines to generate knowledge articles to help learners improve their knowledge about a given topic.

# Your Inputs
The topic of the knowledge article, along with any specific details that the learner desires to know about the topic.

# Your Output
An outline that covers all of the details (at a high level) and is structured as a list of sections.

# Output Format
- You will use XML format for the outline.
- The outermost element will be a <knowledge> element.
- The <knowledge> element will contain a <topic> element that contains the exact string "{{topic_input}}".
- The <knowledge> element will contains <sections> element that contains a list of <section> elements.
- Each <section> element will contain a <title> element that contains the title of the section.
- Each <section> element will contain a <content> element that contains the content of the section.
- The structure of the XML should be as follows:

{{outline_template}}

- ONLY output the XML content of your outline. Do not include any preamble or other text. I'll be parsing this with an XML parser, so if you include any text outside of the XML, the parser will throw an error.
"""


def construct_outliner_system_prompt() -> str:
    outliner_system_template = OUTLINE_TEMPLATE.replace(
        "{{section_title}}", "<!-- The title of the section -->"
    )
    outliner_system_template = outliner_system_template.replace(
        "{{section_content}}",
        "<!-- The recommended content for the section, which should just be plain English describing the types of things this section should cover. -->",
    )
    outliner_system_template = outliner_system_template.replace(
        "{{additional_sections_note}}",
        "<!-- Repeat the above section element for each section you want to create -->",
    )
    return OUTLINER_SYSTEM_PROMPT.replace("{{outline_template}}", outliner_system_template)


def construct_outliner_user_prompt(topic_input: str) -> str:
    return f"""
Here is the topic I'm interested in learning about, along with some specific details:

{topic_input}
"""


KNOWLEDGE_GENERATOR_SYSTEM_PROMPT = """
You are an expert at creating knowledge articles for a given topic. You will be given an outline of a topic with a list of sections that should be covered. Each section represents a high-level overview of the content that should be in the article. You will also be given the name of the title of the section that you are creating content for.

# Your Inputs
- The outline of the knowledge article (in XML format), which will contain a list of sections.
- The name of the section that you are creating content for.

# Input Format

Outline:

{{outline_template}}

Your Section Title:

<title><!-- The title of the section that you are creating content for --></title>

# Your Output
The entire article for your section, taking into account what will already be covered in the other sections. Your goal is for the article to be thorough, helpful, and approachable.

# Output Format
- You will use markdown format for the article.
- You will include the title of your section as a header at the top of the article.
- The title's header will be formatted as a level 2 header (## Title Case).
- Do not include any links in the article.
- Do not include any images in the article.
- ONLY output the content of your article. Do not include any preamble or other text. I'll be parsing this with a Markdown parser, so if you include any text outside of the markdown, the parser will throw an error.
"""


def construct_knowledge_generator_system_prompt() -> str:
    knowledge_generator_system_template = OUTLINE_TEMPLATE.replace(
        "{{topic_input}}", "<!-- The topic that the articles will cover -->"
    )
    knowledge_generator_system_template = knowledge_generator_system_template.replace(
        "{{section_title}}", "<!-- The title of a section -->"
    )
    knowledge_generator_system_template = knowledge_generator_system_template.replace(
        "{{section_content}}", "!-- The recommended content for the section. -->"
    )
    knowledge_generator_system_template = knowledge_generator_system_template.replace(
        "{{additional_sections_note}}", "<!-- Additional sections -->"
    )
    return KNOWLEDGE_GENERATOR_SYSTEM_PROMPT.replace(
        "{{outline_template}}", knowledge_generator_system_template
    )


def construct_knowledge_generator_user_prompt(section_title: str, outline: str) -> str:
    return f"""
Outline:

{outline}

Your Section Title:

<title>{section_title}</title>
"""
