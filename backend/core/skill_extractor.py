"""
skill_extractor.py
==================

Module 2: Extract skills from issue text
"""

class SkillExtractor:

    SKILLS = [
        "Python",
        "JavaScript",
        "Java",
        "C++",
        "SQL",
        "PostgreSQL",
        "MySQL",
        "React",
        "Angular",
        "Flask",
        "Django",
        "Docker",
        "Kubernetes",
        "HTML",
        "CSS",
        "TensorFlow",
        "PyTorch",
        "Machine Learning"
    ]

    def extract(self, text):
        """
        Extract skills from issue text
        """

        text = text.lower()

        found_skills = []

        for skill in self.SKILLS:
            if skill.lower() in text:
                found_skills.append(skill)

        return list(set(found_skills))