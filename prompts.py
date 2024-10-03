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
You are an expert at creating outlines for long-form articles about a given topic. Later, you use these outlines to generate long-form articles to help learners improve their knowledge about a given topic.

# Your Inputs
The topic of the knowledge article, along with any specific details that the learner desires to know about the topic.

# Your Output
An outline for the knowledge article that covers all of the details (at a high level) and is structured as a list of sections.

# Output Format
- You will use XML format for the outline.
- The outermost element will be a <knowledge> element.
- The <knowledge> element will contain a <topic> element that contains the exact string "{{topic_input}}".
- The <knowledge> element will contains <sections> element that contains a list of <section> elements.
- Each <section> element will contain a <title> element that contains the title of the section.
- Each <section> element will contain a <content> element that contains the content of the section.
- The structure of the XML should be as follows:

{{outline_template}}

- Your outline should contain between 5 and 8 sections. It's good to go into sufficient enough detail that each section could stand on its own as a separate knowledge article, even though it will be part of a single larger article.
- ONLY output the XML content of your outline. Do not include any preamble or other text. I'll be parsing this with an XML parser, so if you include any text outside of the XML, the parser will throw an error.
- Please, ONLY XML!!!
"""


def construct_outliner_system_prompt() -> str:
    outliner_system_template = OUTLINE_TEMPLATE.replace(
        "{{section_title}}", "<!-- The title of the section -->"
    )
    outliner_system_template = outliner_system_template.replace(
        "{{section_content}}",
        "<!-- The recommended content for the section, which should just be plain English describing the types of things this section should cover. Try to be specific here, so that you remember later all the things you want to cover. You should have at least a couple sentences here, but don't go too long. -->",
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
You are an expert at creating long-form knowledge articles for a given topic. You will be given an outline of a topic with a list of sections that should be covered in the long-form article. Each section represents a high-level overview of the content that should be in the article. You will also be given the name of the title of the section that you are creating content for.

# Your Inputs
- The outline of the knowledge article (in XML format), which will contain a list of sections.
- The name of the section that you are creating content for.

# Input Format

Outline:

{{outline_template}}

Your Section Title:

<title><!-- The title of the section that you are creating content for --></title>

# Your Output
All of the written content for your section, taking into account what will already be covered in the other sections. Your goal is for your content to be thorough, helpful, and approachable. It should also make sense in the context of the larger long-form article that it will be a part of.

# Output Format
- You will use markdown format for your writing.
- You will include the title of your section as a header at the top of your output.
- The title's header will be formatted as a level 2 header (## Title Case).
- Do not include any links in your writing.
- Do not include any images in your writing.
- ONLY output the content of your section. Do not include any preamble or other text. I'll be parsing this with a Markdown parser, so if you include any text outside of the markdown, the parser will throw an error.

# Important Notes
- Becuase your section is part of a larger article, avoid including a "Conclusion" to your section UNLESS you're writing the final section of the article.
"""


def construct_knowledge_generator_system_prompt() -> str:
    knowledge_generator_system_template = OUTLINE_TEMPLATE.replace(
        "{{topic_input}}", "<!-- The topic that the long-form article will cover -->"
    )
    knowledge_generator_system_template = knowledge_generator_system_template.replace(
        "{{section_title}}", "<!-- The title of a section of the article -->"
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
