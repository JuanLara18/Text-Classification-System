#!/usr/bin/env python3
"""
Test Data Generator for AI Text Classification System
Creates realistic job posting data for testing classification functionality.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class JobDataGenerator:
    """Generate realistic job posting data for testing."""
    
    def __init__(self):
        # Job categories and their typical roles
        self.job_categories = {
            "Software Engineering": [
                "Software Engineer", "Frontend Developer", "Backend Developer", 
                "Full Stack Developer", "DevOps Engineer", "Software Architect",
                "Mobile Developer", "Web Developer", "Systems Engineer"
            ],
            "Data Science & Analytics": [
                "Data Scientist", "Data Analyst", "Machine Learning Engineer",
                "Business Intelligence Analyst", "Data Engineer", "Research Scientist",
                "Statistician", "Analytics Manager"
            ],
            "Product Management": [
                "Product Manager", "Senior Product Manager", "Product Owner",
                "Product Marketing Manager", "Technical Product Manager",
                "VP of Product", "Product Analyst"
            ],
            "Sales & Marketing": [
                "Sales Representative", "Account Manager", "Marketing Manager",
                "Digital Marketing Specialist", "Sales Manager", "Business Development",
                "Content Marketing Manager", "SEO Specialist", "Sales Director"
            ],
            "Human Resources": [
                "HR Manager", "Recruiter", "HR Business Partner", "Talent Acquisition",
                "HR Generalist", "Compensation Analyst", "HR Director",
                "People Operations Manager"
            ],
            "Finance & Accounting": [
                "Financial Analyst", "Accountant", "Finance Manager", "Controller",
                "Investment Analyst", "Tax Specialist", "Audit Manager",
                "Treasury Analyst", "CFO"
            ],
            "Operations & Administration": [
                "Operations Manager", "Administrative Assistant", "Office Manager",
                "Supply Chain Manager", "Facilities Manager", "Executive Assistant",
                "Project Coordinator", "Operations Analyst"
            ],
            "Customer Service": [
                "Customer Support Representative", "Customer Success Manager",
                "Technical Support Specialist", "Call Center Agent",
                "Customer Experience Manager", "Support Team Lead"
            ]
        }
        
        # Skills by category
        self.skills_by_category = {
            "Software Engineering": [
                "Python", "Java", "JavaScript", "React", "Node.js", "SQL", "Git",
                "AWS", "Docker", "Kubernetes", "MongoDB", "PostgreSQL", "REST APIs",
                "Agile", "Scrum", "CI/CD", "Linux", "Microservices"
            ],
            "Data Science & Analytics": [
                "Python", "R", "SQL", "Machine Learning", "Statistics", "Pandas",
                "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Tableau", "PowerBI",
                "Excel", "Jupyter", "Data Visualization", "Big Data", "Hadoop", "Spark"
            ],
            "Product Management": [
                "Product Strategy", "Roadmap Planning", "Market Research", "Analytics",
                "A/B Testing", "User Research", "Agile", "Scrum", "Jira", "Confluence",
                "SQL", "Data Analysis", "Wireframing", "Product Launch"
            ],
            "Sales & Marketing": [
                "CRM", "Salesforce", "Lead Generation", "Cold Calling", "Email Marketing",
                "Social Media", "Google Analytics", "SEO", "SEM", "Content Marketing",
                "Marketing Automation", "HubSpot", "Negotiation", "Presentation"
            ],
            "Human Resources": [
                "Recruiting", "Talent Acquisition", "HRIS", "Workday", "Benefits Administration",
                "Performance Management", "Employee Relations", "Compensation",
                "Training & Development", "Compliance", "Employment Law"
            ],
            "Finance & Accounting": [
                "Excel", "QuickBooks", "SAP", "Financial Modeling", "Budgeting",
                "Forecasting", "GAAP", "Financial Reporting", "Auditing", "Tax Preparation",
                "Bloomberg", "SQL", "Python", "VBA"
            ],
            "Operations & Administration": [
                "Project Management", "Process Improvement", "Supply Chain", "Inventory Management",
                "Vendor Management", "Data Entry", "Microsoft Office", "Scheduling",
                "Customer Service", "Administrative Support"
            ],
            "Customer Service": [
                "Customer Support", "Phone Support", "Live Chat", "Ticketing Systems",
                "CRM", "Problem Solving", "Communication", "Conflict Resolution",
                "Product Knowledge", "Multi-tasking"
            ]
        }
        
        # Experience levels
        self.experience_levels = ["Entry Level", "Mid Level", "Senior Level", "Executive"]
        
        # Company types
        self.company_types = [
            "Technology Startup", "Fortune 500 Company", "Healthcare Organization",
            "Financial Services", "E-commerce Company", "Consulting Firm",
            "Manufacturing Company", "Non-profit Organization", "Government Agency"
        ]
        
        # Locations
        self.locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
            "Boston, MA", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
            "Atlanta, GA", "Remote", "Hybrid - San Francisco", "Hybrid - New York"
        ]
    
    def generate_job_description(self, category, title, skills):
        """Generate a realistic job description."""
        templates = {
            "Software Engineering": [
                f"We are seeking a {title} to join our dynamic engineering team. You will be responsible for developing scalable software solutions, collaborating with cross-functional teams, and contributing to our technology stack. Experience with {', '.join(skills[:3])} is required.",
                f"Join our team as a {title}! You'll work on cutting-edge projects, write clean and maintainable code, and help build products used by millions. Strong knowledge of {', '.join(skills[:3])} is essential.",
                f"Exciting opportunity for a {title} to work on innovative software products. You'll be involved in the full development lifecycle, from design to deployment. Proficiency in {', '.join(skills[:3])} required."
            ],
            "Data Science & Analytics": [
                f"We're looking for a {title} to help us make data-driven decisions. You'll analyze large datasets, build predictive models, and present insights to stakeholders. Experience with {', '.join(skills[:3])} is crucial.",
                f"Join our data team as a {title}! You'll work with big data, create visualizations, and develop machine learning models. Strong skills in {', '.join(skills[:3])} required.",
                f"Seeking a {title} to turn data into actionable insights. You'll work with various data sources, perform statistical analysis, and communicate findings effectively. Proficiency in {', '.join(skills[:3])} needed."
            ],
            "Product Management": [
                f"We need a {title} to drive our product strategy and roadmap. You'll work closely with engineering and design teams, conduct market research, and ensure successful product launches. Experience with {', '.join(skills[:3])} preferred.",
                f"Exciting role for a {title} to shape our product vision. You'll define requirements, prioritize features, and work with stakeholders across the organization. Knowledge of {', '.join(skills[:3])} valuable.",
                f"Join as a {title} and lead product development from conception to launch. You'll analyze user feedback, coordinate with development teams, and drive product growth. Skills in {', '.join(skills[:3])} beneficial."
            ],
            "Sales & Marketing": [
                f"Looking for a {title} to drive revenue growth and expand our customer base. You'll develop sales strategies, build relationships with clients, and achieve ambitious targets. Experience with {', '.join(skills[:3])} required.",
                f"Join our team as a {title}! You'll create compelling marketing campaigns, generate leads, and support sales efforts. Proficiency in {', '.join(skills[:3])} essential.",
                f"We're seeking a {title} to accelerate our business growth. You'll identify new opportunities, manage customer relationships, and contribute to our sales strategy. Knowledge of {', '.join(skills[:3])} important."
            ]
        }
        
        # Get template for category or use generic
        category_templates = templates.get(category, [
            f"We are hiring a {title} to join our growing team. You will contribute to key initiatives and help drive our mission forward. Experience with {', '.join(skills[:3])} preferred.",
            f"Exciting opportunity for a {title} in a fast-paced environment. You'll work on important projects and collaborate with talented professionals. Skills in {', '.join(skills[:3])} valuable."
        ])
        
        return random.choice(category_templates)
    
    def generate_dataset(self, n_samples=1000):
        """Generate complete job dataset."""
        print(f"Generating {n_samples} job postings...")
        
        data = []
        
        for i in range(n_samples):
            # Select category and job title
            category = random.choice(list(self.job_categories.keys()))
            title = random.choice(self.job_categories[category])
            
            # Select skills (3-6 skills per job)
            available_skills = self.skills_by_category[category]
            num_skills = random.randint(3, 6)
            skills = random.sample(available_skills, min(num_skills, len(available_skills)))
            
            # Generate other attributes
            experience = random.choice(self.experience_levels)
            company_type = random.choice(self.company_types)
            location = random.choice(self.locations)
            
            # Generate description
            description = self.generate_job_description(category, title, skills)
            
            # Add some variation to titles
            title_variations = [
                title,
                f"Senior {title}",
                f"Junior {title}",
                f"Lead {title}",
                f"{title} II",
                f"{title} III"
            ]
            
            if "Senior" not in title and random.random() < 0.3:
                title = random.choice(title_variations[1:])
            elif "Junior" not in title and random.random() < 0.15:
                title = title_variations[2]
            
            # Create salary range based on experience and category
            base_salary = {
                "Software Engineering": 85000,
                "Data Science & Analytics": 90000,
                "Product Management": 95000,
                "Sales & Marketing": 65000,
                "Human Resources": 60000,
                "Finance & Accounting": 70000,
                "Operations & Administration": 55000,
                "Customer Service": 45000
            }.get(category, 60000)
            
            experience_multiplier = {
                "Entry Level": 1.0,
                "Mid Level": 1.4,
                "Senior Level": 1.8,
                "Executive": 2.5
            }.get(experience, 1.0)
            
            salary = int(base_salary * experience_multiplier * (0.8 + random.random() * 0.4))
            
            # Add posting date (last 6 months)
            days_ago = random.randint(0, 180)
            posting_date = datetime.now() - timedelta(days=days_ago)
            
            # Create record
            record = {
                'job_id': f"JOB_{i+1:04d}",
                'position_title': title,
                'job_description': description,
                'category': category,
                'experience_level': experience,
                'skills_required': ', '.join(skills),
                'company_type': company_type,
                'location': location,
                'salary_range': f"${salary:,} - ${int(salary * 1.2):,}",
                'posting_date': posting_date.strftime('%Y-%m-%d'),
                'employment_type': random.choice(['Full-time', 'Part-time', 'Contract', 'Internship']),
                'remote_option': random.choice(['On-site', 'Remote', 'Hybrid']),
                'education_required': random.choice(['High School', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD']),
                'years_experience': random.randint(0, 15),
                'industry': random.choice(['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing', 'Education'])
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def save_to_stata(self, df, filename):
        """Save DataFrame to Stata format."""
        print(f"Saving dataset to {filename}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Clean string columns for Stata compatibility
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
            # Limit string length for Stata
            df[col] = df[col].str[:2045]  # Stata has string limits
        
        # Save to Stata format
        df.to_stata(filename, write_index=False, version=117)
        print(f"Dataset saved successfully with {len(df)} records")

def main():
    """Generate test dataset."""
    print("=" * 50)
    print("🔧 AI Text Classification System - Test Data Generator")
    print("=" * 50)
    
    # Create generator
    generator = JobDataGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(n_samples=1000)
    
    # Display sample
    print("\n📊 Sample of generated data:")
    print(df[['position_title', 'category', 'experience_level', 'location']].head(10))
    
    # Show category distribution
    print("\n📈 Category Distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
    
    # Save to Stata file
    output_file = "tests/data/test_job_dataset.dta"
    generator.save_to_stata(df, output_file)
    
    # Also save as CSV for easy inspection
    csv_file = "tests/data/test_job_dataset.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV version saved to {csv_file}")
    
    print("\n✅ Test data generation complete!")
    print(f"Generated 1000 realistic job postings for testing")
    print(f"Ready to use with AI Text Classification System")

if __name__ == "__main__":
    main()