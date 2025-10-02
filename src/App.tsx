import React, { useState, useEffect } from 'react';
import { Monitor, Server, Brain, Bone as Drone, DollarSign, Shield, Map, Code, Github, Linkedin, Mail, Phone, Award, Briefcase, GraduationCap, ChevronRight, X } from 'lucide-react';

interface Project {
  id: string;
  name: string;
  description: string;
  tech: string[];
  icon: React.ReactNode;
  color: string;
  details: string;
}

interface Skill {
  category: string;
  items: string[];
  color: string;
}

interface Experience {
  title: string;
  company: string;
  duration: string;
  description: string;
  icon: React.ReactNode;
}

function App() {
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  const projects: Project[] = [
    {
      id: 'darshan',
      name: 'DARSHAN',
      description: 'CNN Model (U-Net++) for Road/Building Classification',
      tech: ['Python', 'TensorFlow', 'OpenCV', 'U-Net++'],
      icon: <Drone className="w-8 h-8" />,
      color: 'from-emerald-500 to-teal-600',
      details: 'Advanced computer vision model using U-Net++ architecture for precise segmentation of roads and buildings from drone imagery. Achieved high accuracy in urban planning applications.'
    },
    {
      id: 'llm-api',
      name: 'LLM Automation API',
      description: 'RAG-powered Natural Language Command System',
      tech: ['Python', 'LLMs', 'FAISS', 'RAG', 'NLP'],
      icon: <Brain className="w-8 h-8" />,
      color: 'from-purple-500 to-indigo-600',
      details: 'Sophisticated automation system leveraging Large Language Models with Retrieval-Augmented Generation (RAG) and FAISS vector database for natural language command processing.'
    },
    {
      id: 'expensense',
      name: 'ExpenSense',
      description: 'Full-Stack MERN Financial Management System',
      tech: ['React', 'Node.js', 'Express.js', 'MongoDB'],
      icon: <DollarSign className="w-8 h-8" />,
      color: 'from-green-500 to-emerald-600',
      details: 'Comprehensive financial management platform built with MERN stack, featuring expense tracking, budget planning, and intelligent financial insights.'
    },
    {
      id: 'sahyog',
      name: 'Sahyog',
      description: 'Real-time Disaster Resource Tracking System',
      tech: ['React', 'Node.js', 'Socket.io', 'MongoDB'],
      icon: <Map className="w-8 h-8" />,
      color: 'from-red-500 to-orange-600',
      details: 'Real-time disaster management system enabling efficient coordination of resources during emergencies with live tracking and communication features.'
    },
    {
      id: 'cyberbullying',
      name: 'AI Text Analysis',
      description: 'Cyberbullying Detection & Malayalam OCR',
      tech: ['PyTorch', 'Transformers', 'NLP', 'OCR'],
      icon: <Shield className="w-8 h-8" />,
      color: 'from-blue-500 to-cyan-600',
      details: 'Advanced AI models for cyberbullying detection and optical character recognition for Malayalam text using transformer architectures.'
    }
  ];

  const skills: Skill[] = [
    {
      category: 'Programming',
      items: ['Python', 'C++', 'Java', 'JavaScript', 'PHP'],
      color: 'text-cyan-400'
    },
    {
      category: 'AI/ML',
      items: ['TensorFlow', 'Keras', 'PyTorch', 'NLP', 'LLMs'],
      color: 'text-purple-400'
    },
    {
      category: 'Web Dev',
      items: ['React', 'Node.js', 'Express.js', 'HTML', 'CSS'],
      color: 'text-green-400'
    },
    {
      category: 'Database',
      items: ['MongoDB', 'MySQL', 'SQL', 'FAISS'],
      color: 'text-blue-400'
    },
    {
      category: 'DevOps',
      items: ['AWS', 'Docker', 'Git'],
      color: 'text-orange-400'
    }
  ];

  const experiences: Experience[] = [
    {
      title: 'Machine Learning Intern',
      company: 'NIT Calicut',
      duration: '2023',
      description: 'Developed advanced ML models for computer vision applications',
      icon: <Brain className="w-5 h-5" />
    },
    {
      title: 'Cybersecurity Intern',
      company: 'IBM',
      duration: '2023',
      description: 'Worked on security protocols and threat detection systems',
      icon: <Shield className="w-5 h-5" />
    },
    {
      title: 'Data Science Intern',
      company: 'Coratia Technologies',
      duration: '2022',
      description: 'Analyzed complex datasets and built predictive models',
      icon: <Award className="w-5 h-5" />
    }
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="w-20 h-20 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <div className="w-16 h-16 border-4 border-purple-400 border-b-transparent rounded-full animate-spin absolute top-2 left-1/2 transform -translate-x-1/2"></div>
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Initializing AI Lab</h2>
          <p className="text-cyan-400">Loading Ashmi's Portfolio...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-blue-900 to-slate-900 overflow-x-hidden">
      {/* Floating Particles Background */}
      <div className="fixed inset-0 pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full animate-ping"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 2}s`
            }}
          />
        ))}
      </div>

      {/* Header */}
      <header className="relative z-10 bg-slate-900/80 backdrop-blur-md border-b border-cyan-400/20">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">Ashmi S N</h1>
              <p className="text-cyan-400">Full-Stack Developer & AI/ML Enthusiast</p>
            </div>
            <div className="flex space-x-4">
              <a href="https://github.com" className="text-gray-400 hover:text-cyan-400 transition-colors">
                <Github className="w-6 h-6" />
              </a>
              <a href="https://linkedin.com" className="text-gray-400 hover:text-cyan-400 transition-colors">
                <Linkedin className="w-6 h-6" />
              </a>
              <a href="mailto:ashmi@example.com" className="text-gray-400 hover:text-cyan-400 transition-colors">
                <Mail className="w-6 h-6" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h2 className="text-5xl md:text-7xl font-bold text-white mb-6">
            AI <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600">LAB</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Welcome to my digital workspace where innovation meets implementation. 
            Explore my journey through artificial intelligence, full-stack development, and cutting-edge technology.
          </p>
        </div>

        {/* Main Workstation */}
        <div className="grid lg:grid-cols-3 gap-8 mb-16">
          {/* Projects Monitor */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 backdrop-blur-md border border-cyan-400/20 rounded-xl p-8 hover:border-cyan-400/40 transition-all duration-300 transform hover:scale-[1.02]">
              <div className="flex items-center mb-6">
                <Monitor className="w-8 h-8 text-cyan-400 mr-3" />
                <h3 className="text-2xl font-bold text-white">Featured Projects</h3>
              </div>
              <div className="grid md:grid-cols-2 gap-6">
                {projects.map((project) => (
                  <div
                    key={project.id}
                    className="group cursor-pointer"
                    onClick={() => setSelectedProject(project)}
                  >
                    <div className={`bg-gradient-to-r ${project.color} p-1 rounded-lg mb-4 transform group-hover:scale-105 transition-all duration-300`}>
                      <div className="bg-slate-900 p-6 rounded-lg h-full">
                        <div className="flex items-center mb-3">
                          <div className="text-white mr-3">
                            {project.icon}
                          </div>
                          <h4 className="text-lg font-semibold text-white">{project.name}</h4>
                        </div>
                        <p className="text-gray-300 text-sm mb-3">{project.description}</p>
                        <div className="flex flex-wrap gap-2">
                          {project.tech.slice(0, 3).map((tech) => (
                            <span key={tech} className="text-xs bg-slate-700 text-cyan-400 px-2 py-1 rounded">
                              {tech}
                            </span>
                          ))}
                          {project.tech.length > 3 && (
                            <span className="text-xs text-gray-400">+{project.tech.length - 3} more</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Server Rack - Skills */}
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-md border border-purple-400/20 rounded-xl p-6 hover:border-purple-400/40 transition-all duration-300">
              <div className="flex items-center mb-4">
                <Server className="w-6 h-6 text-purple-400 mr-3" />
                <h3 className="text-xl font-bold text-white">Tech Stack</h3>
              </div>
              <div className="space-y-4">
                {skills.map((skill) => (
                  <div key={skill.category} className="group">
                    <h4 className={`font-semibold ${skill.color} mb-2`}>{skill.category}</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {skill.items.map((item) => (
                        <div key={item} className="bg-slate-700/50 px-3 py-1 rounded text-sm text-gray-300 hover:bg-slate-600/50 transition-colors">
                          {item}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Experience Hologram */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-green-400/20 rounded-xl p-8 mb-16 hover:border-green-400/40 transition-all duration-300">
          <div className="flex items-center mb-6">
            <Briefcase className="w-8 h-8 text-green-400 mr-3" />
            <h3 className="text-2xl font-bold text-white">Professional Experience</h3>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {experiences.map((exp, index) => (
              <div key={index} className="relative group">
                <div className="bg-gradient-to-b from-slate-700/50 to-slate-800/50 border border-gray-600 rounded-lg p-6 hover:border-green-400/50 transition-all duration-300 transform hover:scale-105">
                  <div className="flex items-center mb-3">
                    <div className="bg-green-400/20 p-2 rounded-lg mr-3">
                      {exp.icon}
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-white">{exp.title}</h4>
                      <p className="text-green-400 text-sm">{exp.company}</p>
                    </div>
                  </div>
                  <p className="text-gray-300 text-sm mb-2">{exp.description}</p>
                  <span className="text-xs text-gray-400">{exp.duration}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Contact Console */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-orange-400/20 rounded-xl p-8 hover:border-orange-400/40 transition-all duration-300">
          <div className="flex items-center mb-6">
            <Code className="w-8 h-8 text-orange-400 mr-3" />
            <h3 className="text-2xl font-bold text-white">Initiate Connection</h3>
          </div>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <p className="text-gray-300 mb-6">
                Ready to collaborate on innovative projects? Let's build something amazing together.
              </p>
              <div className="space-y-4">
                <div className="flex items-center">
                  <Mail className="w-5 h-5 text-orange-400 mr-3" />
                  <span className="text-gray-300">ashmi.sn@email.com</span>
                </div>
                <div className="flex items-center">
                  <Phone className="w-5 h-5 text-orange-400 mr-3" />
                  <span className="text-gray-300">+91 XXXXX XXXXX</span>
                </div>
              </div>
            </div>
            <div className="flex flex-col justify-center">
              <button className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white px-8 py-3 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 mb-4">
                Download Resume
              </button>
              <button className="border border-orange-400 text-orange-400 hover:bg-orange-400 hover:text-white px-8 py-3 rounded-lg font-semibold transition-all duration-300">
                Schedule Meeting
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Project Modal */}
      {selectedProject && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
          <div className="bg-slate-800 border border-cyan-400/30 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-8">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <div className={`bg-gradient-to-r ${selectedProject.color} p-3 rounded-lg mr-4`}>
                    {selectedProject.icon}
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white">{selectedProject.name}</h3>
                    <p className="text-gray-400">{selectedProject.description}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedProject(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
              
              <div className="space-y-6">
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Project Details</h4>
                  <p className="text-gray-300 leading-relaxed">{selectedProject.details}</p>
                </div>
                
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Technologies Used</h4>
                  <div className="flex flex-wrap gap-3">
                    {selectedProject.tech.map((tech) => (
                      <span key={tech} className="bg-slate-700 text-cyan-400 px-4 py-2 rounded-lg text-sm font-medium">
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="flex space-x-4 pt-4">
                  <button className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105">
                    View Live Demo
                  </button>
                  <button className="flex-1 border border-cyan-400 text-cyan-400 hover:bg-cyan-400 hover:text-white px-6 py-3 rounded-lg font-semibold transition-all duration-300">
                    Source Code
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;