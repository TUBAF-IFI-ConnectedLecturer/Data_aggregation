import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict

from pipeline.taskfactory import TaskWithInputFileMonitor
from pipeline.taskfactory import loggedExecution


class AnalyzeFeatureClusters(TaskWithInputFileMonitor):
    """
    Analyze LiaScript document clusters based on feature usage patterns.

    This stage:
    1. Creates feature vectors for each document
    2. Identifies user segments/document types based on feature combinations
    3. Calculates feature co-occurrence matrix
    4. Provides insights for documentation gaps and feature prioritization

    Document types identified:
    - MINT-Author: Math-heavy, uses TikZ-Jax, Algebrite, code blocks
    - Presenter: Animation-focused, TTS, images, few quizzes
    - Assessment-Focus: High quiz density, hints, MC+SC
    - Multimedia-Course: Video, audio, webapps, images
    - Template-Power-User: Many imports, external scripts
    - Minimalist: Few features, mainly text
    """

    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])

        self.feature_analysis_name = Path(config_global['raw_data_folder']) / stage_param['feature_analysis_name']
        self.cluster_output_name = Path(config_global['raw_data_folder']) / stage_param['cluster_output_name']
        self.force_run = stage_param.get('force_run', False)

        # Feature groups for clustering
        self.feature_groups = {
            'mint': ['feature:has_math', 'feature:has_code_blocks', 'feature:has_lia_viz_tables'],
            'presentation': ['feature:has_animation_fragments', 'feature:has_narrator', 'feature:has_images'],
            'assessment': ['feature:has_quiz', 'feature:has_quiz_hints', 'feature:has_single_choice', 'feature:has_multiple_choice'],
            'multimedia': ['feature:has_video', 'feature:has_audio', 'feature:has_webapp', 'feature:has_images'],
            'advanced': ['feature:has_imports', 'feature:has_external_scripts', 'feature:has_macros'],
            'interactive': ['feature:has_script_tags', 'feature:has_effects', 'feature:has_classroom']
        }

        # Official LiaTemplates categorization
        self.official_templates = {
            # Math & Science
            'tikz-jax': 'math_visualization',
            'algebrite': 'symbolic_math',
            'jsxgraph': 'math_interactive',
            # Visualization
            'ggbscript': 'geogebra',
            'mermaid_template': 'diagrams',
            'plantuml': 'diagrams',
            'vtk': '3d_visualization',
            'beforeandafter': 'comparison',
            # Code Execution
            'pyodide': 'python_execution',
            'coderunner': 'code_execution',
            'rextester': 'online_compiler',
            'avr8js': 'microcontroller',
            'jscpp': 'cpp_execution',
            'jscad': '3d_modeling',
            'pyscript': 'python_execution',
            'biwascheme': 'scheme_execution',
            'tiny-turtle': 'turtle_graphics',
            'processingjs': 'creative_coding',
            # Domain-specific
            'kekulejs': 'chemistry',
            'abcjs': 'music_notation',
            'textanalysis': 'nlp',
            'citations': 'bibliography',
            'alasql': 'database',
            # Interactive/UI
            'collaborativedrawing': 'collaborative',
            'webdev': 'web_development',
            'speech-recognition-quiz': 'speech_recognition',
            'netswarm-simulator': 'networking',
            'material-design-icons': 'ui_icons',
            'fullscreen': 'presentation_mode',
            'random': 'utilities',
        }

        self.template_categories = {
            'math_science': ['tikz-jax', 'algebrite', 'jsxgraph', 'ggbscript'],
            'code_execution': ['pyodide', 'coderunner', 'rextester', 'avr8js', 'jscpp', 'pyscript', 'biwascheme', 'tiny-turtle', 'processingjs'],
            'visualization': ['mermaid_template', 'plantuml', 'vtk', 'beforeandafter', 'jscad'],
            'domain_specific': ['kekulejs', 'abcjs', 'textanalysis', 'citations', 'alasql'],
            'interactive': ['collaborativedrawing', 'webdev', 'speech-recognition-quiz', 'netswarm-simulator'],
        }

    @loggedExecution
    def execute_task(self):
        if not Path(self.feature_analysis_name).exists():
            logging.error(f"Feature analysis file not found: {self.feature_analysis_name}")
            return

        df_features = pd.read_pickle(Path(self.feature_analysis_name))
        logging.info(f"Loaded {len(df_features)} documents for clustering analysis")

        # Calculate cluster assignments
        cluster_data = self._assign_clusters(df_features)

        # Analyze official LiaTemplates usage
        template_analysis = self._analyze_official_templates(df_features)

        # Add template info to cluster profiles
        cluster_data['profiles'] = self._add_template_to_cluster_profiles(
            cluster_data['profiles'],
            df_features,
            cluster_data['assignments'],
            template_analysis
        )

        # Calculate feature co-occurrence matrix
        cooccurrence = self._calculate_cooccurrence(df_features)

        # Calculate complexity metrics
        complexity = self._calculate_complexity(df_features)

        # Generate documentation gap analysis
        doc_gaps = self._analyze_documentation_gaps(df_features)

        # Analyze by author (to avoid single-author bias)
        author_analysis = self._analyze_by_author(df_features, template_analysis)

        # Compile results
        results = {
            'cluster_assignments': cluster_data['assignments'],
            'cluster_statistics': cluster_data['statistics'],
            'cluster_profiles': cluster_data['profiles'],
            'official_templates': template_analysis,
            'author_analysis': author_analysis,
            'cooccurrence_matrix': cooccurrence,
            'complexity_metrics': complexity,
            'documentation_gaps': doc_gaps,
            'total_documents': len(df_features),
            'total_authors': author_analysis['total_authors'],
            'timestamp': pd.Timestamp.now()
        }

        # Save results
        pd.to_pickle(results, self.cluster_output_name)

        # Save readable report
        self._save_report(results)

        logging.info(f"Cluster analysis saved to {self.cluster_output_name}")

    def _assign_clusters(self, df):
        """Assign documents to clusters based on feature patterns."""
        assignments = {}
        cluster_counts = defaultdict(int)

        # Define cluster criteria
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Assigning clusters"):
            doc_id = row.get('pipe:ID', idx)
            clusters = []

            # MINT-Author: Math + Code + Viz Tables
            mint_score = sum([
                row.get('feature:has_math', False),
                row.get('feature:has_code_blocks', False),
                row.get('feature:has_lia_viz_tables', False)
            ])
            if mint_score >= 2:
                clusters.append('mint_author')

            # Presenter: Animation + TTS + Images, low quiz
            presenter_score = sum([
                row.get('feature:has_animation_fragments', False),
                row.get('feature:has_narrator', False),
                row.get('feature:has_images', False)
            ])
            quiz_count = row.get('feature:total_quiz_count', 0)
            if presenter_score >= 2 and quiz_count < 5:
                clusters.append('presenter')

            # Assessment-Focus: High quiz density
            has_quiz = row.get('feature:has_quiz', False)
            has_hints = row.get('feature:has_quiz_hints', False)
            has_mc = row.get('feature:has_multiple_choice', False) or row.get('feature:has_single_choice', False)
            if has_quiz and (has_hints or has_mc):
                clusters.append('assessment_focus')

            # Multimedia-Course: Video/Audio/Webapp
            multimedia_score = sum([
                row.get('feature:has_video', False),
                row.get('feature:has_audio', False),
                row.get('feature:has_webapp', False)
            ])
            if multimedia_score >= 1:
                clusters.append('multimedia_course')

            # Template-Power-User: Many imports + external scripts
            has_imports = row.get('feature:has_imports', False)
            has_ext_scripts = row.get('feature:has_external_scripts', False)
            has_macros = row.get('feature:has_macros', False)
            if has_imports and (has_ext_scripts or has_macros):
                clusters.append('template_power_user')

            # Minimalist: Few features
            feature_count = sum([
                row.get('feature:has_video', False),
                row.get('feature:has_audio', False),
                row.get('feature:has_quiz', False),
                row.get('feature:has_code_blocks', False),
                row.get('feature:has_math', False),
                row.get('feature:has_animation_fragments', False),
                row.get('feature:has_tables', False)
            ])
            if feature_count <= 1:
                clusters.append('minimalist')

            # Default: general if no specific cluster
            if not clusters:
                clusters.append('general')

            assignments[doc_id] = clusters
            for c in clusters:
                cluster_counts[c] += 1

        # Calculate statistics
        statistics = {}
        total = len(df)
        for cluster, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
            statistics[cluster] = {
                'count': count,
                'percentage': round(count / total * 100, 2)
            }

        # Generate cluster profiles
        profiles = self._generate_cluster_profiles(df, assignments)

        return {
            'assignments': assignments,
            'statistics': statistics,
            'profiles': profiles
        }

    def _generate_cluster_profiles(self, df, assignments):
        """Generate detailed profiles for each cluster."""
        profiles = {}

        # Invert assignments to group documents by cluster
        cluster_docs = defaultdict(list)
        for doc_id, clusters in assignments.items():
            for c in clusters:
                cluster_docs[c].append(doc_id)

        # For each cluster, calculate average feature usage
        feature_cols = [col for col in df.columns if col.startswith('feature:has_')]

        for cluster, doc_ids in cluster_docs.items():
            cluster_df = df[df['pipe:ID'].isin(doc_ids)]
            if len(cluster_df) == 0:
                continue

            # Calculate feature prevalence in this cluster
            feature_prevalence = {}
            for col in feature_cols:
                if col in cluster_df.columns:
                    prevalence = cluster_df[col].sum() / len(cluster_df) * 100
                    feature_prevalence[col.replace('feature:has_', '')] = round(prevalence, 1)

            # Sort by prevalence
            sorted_features = dict(sorted(feature_prevalence.items(), key=lambda x: x[1], reverse=True))

            profiles[cluster] = {
                'document_count': len(cluster_df),
                'top_features': dict(list(sorted_features.items())[:10]),
                'all_features': sorted_features
            }

        return profiles

    def _analyze_official_templates(self, df):
        """Analyze usage of official LiaTemplates per document and cluster."""
        template_analysis = {
            'template_usage': {},
            'category_usage': {},
            'per_document': {}
        }

        # Check if template data exists
        if 'feature:imported_templates' not in df.columns:
            logging.warning("No template data found in features DataFrame")
            return template_analysis

        total_docs = len(df)
        template_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for idx, row in df.iterrows():
            doc_id = row.get('pipe:ID', idx)
            templates = row.get('feature:imported_templates', [])
            if not isinstance(templates, list):
                templates = []

            doc_templates = []
            doc_categories = set()

            for template_url in templates:
                # Extract template name from URL
                template_name = self._extract_template_name(template_url)
                if template_name:
                    # Check if it's an official LiaTemplate
                    if template_name in self.official_templates:
                        template_counts[template_name] += 1
                        category = self.official_templates[template_name]
                        category_counts[category] += 1
                        doc_templates.append(template_name)
                        doc_categories.add(category)

            template_analysis['per_document'][doc_id] = {
                'official_templates': doc_templates,
                'categories': list(doc_categories)
            }

        # Calculate percentages
        for template, count in sorted(template_counts.items(), key=lambda x: x[1], reverse=True):
            template_analysis['template_usage'][template] = {
                'count': count,
                'percentage': round(count / total_docs * 100, 2),
                'category': self.official_templates.get(template, 'unknown')
            }

        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            template_analysis['category_usage'][category] = {
                'count': count,
                'percentage': round(count / total_docs * 100, 2)
            }

        return template_analysis

    def _extract_template_name(self, url):
        """Extract template name from LiaTemplate URL."""
        if not url:
            return None

        url_lower = url.lower()

        # Match patterns like github.com/liatemplates/NAME or raw.githubusercontent.com/liatemplates/NAME
        patterns = [
            r'github\.com/liatemplates/([^/]+)',
            r'raw\.githubusercontent\.com/liatemplates/([^/]+)',
            r'github\.com/liascript/([^/]+)',
            r'raw\.githubusercontent\.com/liascript/([^/]+)',
            r'github\.com/liascript-templates/([^/]+)',
            r'raw\.githubusercontent\.com/liascript-templates/([^/]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_lower)
            if match:
                name = match.group(1)
                # Clean up common suffixes
                name = re.sub(r'_template$', '', name)
                name = re.sub(r'-template$', '', name)
                return name

        return None

    def _analyze_by_author(self, df, template_analysis):
        """
        Analyze template usage by author instead of by document.
        This avoids bias from prolific authors (e.g., one author with 1000 docs).
        """
        author_analysis = {
            'total_authors': 0,
            'author_distribution': {},
            'template_adoption': {},  # How many unique authors use each template
            'category_adoption': {},  # How many unique authors use each category
            'author_profiles': {},    # Top authors with their template usage
            'feature_adoption': {},   # How many unique authors use each feature
            'author_clusters': {},    # K-Means clusters with features + templates
        }

        if 'repo_user' not in df.columns:
            logging.warning("No author (repo_user) column found")
            return author_analysis

        # Group documents by author
        authors = df.groupby('repo_user')
        total_authors = len(authors)
        author_analysis['total_authors'] = total_authors

        # Author size distribution
        author_doc_counts = df['repo_user'].value_counts()
        author_analysis['author_distribution'] = {
            'authors_with_1_doc': int((author_doc_counts == 1).sum()),
            'authors_with_2_10_docs': int(((author_doc_counts >= 2) & (author_doc_counts <= 10)).sum()),
            'authors_with_11_50_docs': int(((author_doc_counts > 10) & (author_doc_counts <= 50)).sum()),
            'authors_with_51_100_docs': int(((author_doc_counts > 50) & (author_doc_counts <= 100)).sum()),
            'authors_with_100plus_docs': int((author_doc_counts > 100).sum()),
        }

        # Template adoption by unique authors
        template_authors = defaultdict(set)
        category_authors = defaultdict(set)

        for author, group in authors:
            for idx, row in group.iterrows():
                doc_id = row.get('pipe:ID', idx)
                if doc_id in template_analysis['per_document']:
                    doc_data = template_analysis['per_document'][doc_id]
                    for template in doc_data['official_templates']:
                        template_authors[template].add(author)
                    for category in doc_data['categories']:
                        category_authors[category].add(author)

        # Convert to adoption counts and percentages
        for template, author_set in sorted(template_authors.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(author_set)
            author_analysis['template_adoption'][template] = {
                'author_count': count,
                'adoption_pct': round(count / total_authors * 100, 2),
                'category': self.official_templates.get(template, 'unknown')
            }

        for category, author_set in sorted(category_authors.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(author_set)
            author_analysis['category_adoption'][category] = {
                'author_count': count,
                'adoption_pct': round(count / total_authors * 100, 2)
            }

        # Feature adoption by unique authors
        feature_cols = [col for col in df.columns if col.startswith('feature:has_')]
        for col in feature_cols:
            if col in df.columns:
                # Count unique authors who have at least one doc with this feature
                authors_with_feature = df[df[col] == True]['repo_user'].nunique()
                feature_name = col.replace('feature:has_', '')
                author_analysis['feature_adoption'][feature_name] = {
                    'author_count': authors_with_feature,
                    'adoption_pct': round(authors_with_feature / total_authors * 100, 2)
                }

        # Sort feature adoption by percentage
        author_analysis['feature_adoption'] = dict(
            sorted(author_analysis['feature_adoption'].items(),
                   key=lambda x: x[1]['adoption_pct'], reverse=True)
        )

        # Top author profiles (top 10 by document count)
        for author in author_doc_counts.head(10).index:
            author_docs = df[df['repo_user'] == author]
            author_templates = set()
            author_categories = set()

            for idx, row in author_docs.iterrows():
                doc_id = row.get('pipe:ID', idx)
                if doc_id in template_analysis['per_document']:
                    doc_data = template_analysis['per_document'][doc_id]
                    author_templates.update(doc_data['official_templates'])
                    author_categories.update(doc_data['categories'])

            author_analysis['author_profiles'][author] = {
                'document_count': len(author_docs),
                'templates_used': list(author_templates),
                'categories_used': list(author_categories),
                'template_count': len(author_templates)
            }

        # Perform K-Means clustering on authors using BOTH features AND templates
        author_analysis['author_clusters'] = self._cluster_authors_with_templates(
            df, template_analysis, author_doc_counts
        )

        return author_analysis

    def _cluster_authors_with_templates(self, df, template_analysis, author_doc_counts):
        """
        Perform K-Means clustering on authors using combined feature + template vectors.

        Features include:
        - Boolean features (has_quiz, has_math, etc.)
        - Official template usage (per author)
        - Template categories (per author)
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Only cluster authors with at least 2 documents (to have meaningful feature usage)
        active_authors = author_doc_counts[author_doc_counts >= 2].index.tolist()
        logging.info(f"Clustering {len(active_authors)} authors with ≥2 documents")

        if len(active_authors) < 10:
            logging.warning("Not enough active authors for meaningful clustering")
            return {}

        # Build author-level feature vectors
        feature_cols = [col for col in df.columns if col.startswith('feature:has_')]
        author_feature_matrix = []
        author_names = []

        # Get all official templates for column creation
        all_templates = list(self.official_templates.keys())
        all_categories = list(set(self.official_templates.values()))

        for author in active_authors:
            author_docs = df[df['repo_user'] == author]
            author_vector = {}
            author_names.append(author)

            # 1. Feature usage (percentage of author's docs that use each feature)
            for col in feature_cols:
                if col in author_docs.columns:
                    feature_name = col.replace('feature:has_', '')
                    usage_pct = author_docs[col].sum() / len(author_docs) * 100
                    author_vector[f'feat:{feature_name}'] = usage_pct

            # 2. Template usage (does this author use each official template?)
            author_templates = set()
            author_categories = set()

            for idx, row in author_docs.iterrows():
                doc_id = row.get('pipe:ID', idx)
                if doc_id in template_analysis['per_document']:
                    doc_data = template_analysis['per_document'][doc_id]
                    author_templates.update(doc_data['official_templates'])
                    author_categories.update(doc_data['categories'])

            # Binary: does author use this template?
            for template in all_templates:
                author_vector[f'tmpl:{template}'] = 100.0 if template in author_templates else 0.0

            # Binary: does author use templates in this category?
            for category in all_categories:
                author_vector[f'cat:{category}'] = 100.0 if category in author_categories else 0.0

            author_feature_matrix.append(author_vector)

        # Convert to DataFrame
        author_df = pd.DataFrame(author_feature_matrix, index=author_names)

        # Get feature names for later reporting
        feature_names = list(author_df.columns)
        n_feat_cols = len([c for c in feature_names if c.startswith('feat:')])
        n_tmpl_cols = len([c for c in feature_names if c.startswith('tmpl:')])
        n_cat_cols = len([c for c in feature_names if c.startswith('cat:')])

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(author_df.values)

        # Determine optimal number of clusters (use 5-7 for interpretability)
        n_clusters = min(6, max(3, len(active_authors) // 20))

        # Perform K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Add cluster labels to DataFrame
        author_df['cluster'] = cluster_labels
        author_df['doc_count'] = [author_doc_counts[author] for author in author_names]

        # Analyze each cluster
        clusters = {}
        cluster_names = self._generate_cluster_names(author_df, feature_names)

        for cluster_id in range(n_clusters):
            cluster_authors = author_df[author_df['cluster'] == cluster_id]

            if len(cluster_authors) == 0:
                continue

            # Top features in this cluster (average usage)
            feature_means = {}
            for col in feature_names:
                if col.startswith('feat:'):
                    feature_means[col.replace('feat:', '')] = cluster_authors[col].mean()

            top_features = dict(sorted(feature_means.items(), key=lambda x: x[1], reverse=True)[:10])

            # Top templates in this cluster
            template_means = {}
            for col in feature_names:
                if col.startswith('tmpl:'):
                    mean_val = cluster_authors[col].mean()
                    if mean_val > 0:  # Only include used templates
                        template_means[col.replace('tmpl:', '')] = mean_val

            top_templates = dict(sorted(template_means.items(), key=lambda x: x[1], reverse=True)[:5])

            # Top categories in this cluster
            category_means = {}
            for col in feature_names:
                if col.startswith('cat:'):
                    mean_val = cluster_authors[col].mean()
                    if mean_val > 0:
                        category_means[col.replace('cat:', '')] = mean_val

            top_categories = dict(sorted(category_means.items(), key=lambda x: x[1], reverse=True)[:5])

            # Representative authors (sorted by doc count)
            rep_authors = cluster_authors.nlargest(5, 'doc_count').index.tolist()

            clusters[cluster_id] = {
                'name': cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
                'author_count': len(cluster_authors),
                'total_docs': int(cluster_authors['doc_count'].sum()),
                'avg_docs_per_author': round(cluster_authors['doc_count'].mean(), 1),
                'top_features': top_features,
                'top_templates': top_templates,
                'top_categories': top_categories,
                'representative_authors': rep_authors,
                'authors': cluster_authors.index.tolist()
            }

        return {
            'n_clusters': n_clusters,
            'n_features_used': len(feature_names),
            'feature_types': f'{n_feat_cols} features, {n_tmpl_cols} templates, {n_cat_cols} categories',
            'authors_clustered': len(active_authors),
            'clusters': clusters
        }

    def _generate_cluster_names(self, author_df, feature_names):
        """Generate descriptive names for clusters based on their characteristics."""
        cluster_names = {}

        for cluster_id in author_df['cluster'].unique():
            cluster_authors = author_df[author_df['cluster'] == cluster_id]

            # Find dominant characteristics
            characteristics = []

            # Check feature dominance
            feat_cols = [c for c in feature_names if c.startswith('feat:')]
            for col in feat_cols:
                if cluster_authors[col].mean() > 70:  # >70% average usage
                    characteristics.append(col.replace('feat:', ''))

            # Check template dominance
            tmpl_cols = [c for c in feature_names if c.startswith('tmpl:')]
            for col in tmpl_cols:
                if cluster_authors[col].mean() > 30:  # >30% use this template
                    characteristics.append(f"uses {col.replace('tmpl:', '')}")

            # Generate name based on characteristics
            if 'quiz' in characteristics and 'quiz_hints' in characteristics:
                cluster_names[cluster_id] = 'Quiz-Experten (Assessments)'
            elif 'math' in characteristics and any('tikz' in c for c in characteristics):
                cluster_names[cluster_id] = 'MINT-Power-User (Math+TikZ)'
            elif 'video' in characteristics and 'audio' in characteristics:
                cluster_names[cluster_id] = 'Multimedia-Produzenten'
            elif 'narrator' in characteristics and 'images' in characteristics:
                cluster_names[cluster_id] = 'Präsentatoren'
            elif 'external_scripts' in characteristics and 'macros' in characteristics:
                cluster_names[cluster_id] = 'Template-Power-User'
            elif any('coderunner' in c for c in characteristics):
                cluster_names[cluster_id] = 'Code-Ausführer (CodeRunner)'
            elif any('pyodide' in c for c in characteristics):
                cluster_names[cluster_id] = 'Python-Enthusiasten'
            elif len(characteristics) <= 2:
                cluster_names[cluster_id] = 'Minimalisten'
            else:
                cluster_names[cluster_id] = f'Cluster {cluster_id}'

        return cluster_names

    def _add_template_to_cluster_profiles(self, profiles, df, assignments, template_analysis):
        """Add official template usage to cluster profiles."""
        # Invert assignments to group documents by cluster
        cluster_docs = defaultdict(list)
        for doc_id, clusters in assignments.items():
            for c in clusters:
                cluster_docs[c].append(doc_id)

        for cluster, doc_ids in cluster_docs.items():
            if cluster not in profiles:
                continue

            # Count template usage in this cluster
            cluster_template_counts = defaultdict(int)
            cluster_category_counts = defaultdict(int)

            for doc_id in doc_ids:
                if doc_id in template_analysis['per_document']:
                    doc_data = template_analysis['per_document'][doc_id]
                    for template in doc_data['official_templates']:
                        cluster_template_counts[template] += 1
                    for category in doc_data['categories']:
                        cluster_category_counts[category] += 1

            cluster_size = profiles[cluster]['document_count']

            # Top templates in cluster
            top_templates = {}
            for template, count in sorted(cluster_template_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                top_templates[template] = {
                    'count': count,
                    'percentage': round(count / cluster_size * 100, 1)
                }

            # Category distribution in cluster
            category_dist = {}
            for category, count in sorted(cluster_category_counts.items(), key=lambda x: x[1], reverse=True):
                category_dist[category] = {
                    'count': count,
                    'percentage': round(count / cluster_size * 100, 1)
                }

            profiles[cluster]['official_templates'] = top_templates
            profiles[cluster]['template_categories'] = category_dist

        return profiles

    def _calculate_cooccurrence(self, df):
        """Calculate feature co-occurrence matrix."""
        feature_cols = [col for col in df.columns if col.startswith('feature:has_')]

        # Create binary feature matrix
        feature_matrix = df[feature_cols].fillna(False).astype(int)

        # Calculate co-occurrence (features that appear together)
        cooccurrence = feature_matrix.T.dot(feature_matrix)

        # Normalize by total documents
        total = len(df)
        cooccurrence_pct = (cooccurrence / total * 100).round(2)

        # Convert to dictionary for serialization
        cooccurrence_dict = {}
        for col in cooccurrence_pct.columns:
            clean_name = col.replace('feature:has_', '')
            cooccurrence_dict[clean_name] = {}
            for idx in cooccurrence_pct.index:
                clean_idx = idx.replace('feature:has_', '')
                cooccurrence_dict[clean_name][clean_idx] = cooccurrence_pct.loc[idx, col]

        return cooccurrence_dict

    def _calculate_complexity(self, df):
        """Calculate complexity metrics for documents."""
        complexity = {}

        # Feature count per document
        feature_cols = [col for col in df.columns if col.startswith('feature:has_')]
        df['_feature_count'] = df[feature_cols].fillna(False).sum(axis=1)

        complexity['avg_features_per_doc'] = round(df['_feature_count'].mean(), 2)
        complexity['median_features_per_doc'] = round(df['_feature_count'].median(), 2)
        complexity['max_features_per_doc'] = int(df['_feature_count'].max())
        complexity['min_features_per_doc'] = int(df['_feature_count'].min())

        # Feature distribution
        complexity['feature_distribution'] = df['_feature_count'].value_counts().sort_index().to_dict()

        # Quiz density (quiz elements per document)
        if 'feature:total_quiz_count' in df.columns:
            complexity['avg_quiz_count'] = round(df['feature:total_quiz_count'].mean(), 2)
            complexity['docs_with_10plus_quizzes'] = int((df['feature:total_quiz_count'] >= 10).sum())

        # Heading structure
        if 'feature:total_headings' in df.columns:
            complexity['avg_headings'] = round(df['feature:total_headings'].mean(), 2)

        # Clean up temporary column
        df.drop('_feature_count', axis=1, inplace=True)

        return complexity

    def _analyze_documentation_gaps(self, df):
        """Analyze potential documentation gaps based on feature usage patterns."""
        gaps = {}

        # Features with low adoption might indicate documentation issues
        feature_cols = [col for col in df.columns if col.startswith('feature:has_')]
        total = len(df)

        for col in feature_cols:
            if col in df.columns:
                usage_pct = df[col].sum() / total * 100
                feature_name = col.replace('feature:has_', '')

                # Low usage features (< 10%)
                if usage_pct < 10:
                    gaps[feature_name] = {
                        'usage_pct': round(usage_pct, 2),
                        'hypothesis': self._get_gap_hypothesis(feature_name, usage_pct)
                    }

        # Sort by usage
        gaps = dict(sorted(gaps.items(), key=lambda x: x[1]['usage_pct']))

        return gaps

    def _get_gap_hypothesis(self, feature_name, usage_pct):
        """Generate hypothesis for why a feature might be underused."""
        hypotheses = {
            'webapp': 'Complex syntax (??[]) not intuitive; documentation may be insufficient',
            'quiz_hints': 'Feature may not be well-known; often overlooked in documentation',
            'lia_viz_tables': 'Requires understanding of data-type annotations; steep learning curve',
            'animated_css': 'Requires CSS knowledge; niche use case',
            'qr_codes': 'Niche application; primarily useful for printed materials',
            'effects': 'Advanced feature; may lack prominent documentation',
            'classroom': 'Collaborative features require setup; niche use case',
            'ascii_diagrams': 'Specialized use case; requires specific tooling knowledge',
            'matrix_quiz': 'Complex quiz type; documentation may be insufficient',
            'surveys': 'Distinct from quizzes; may be confused with single-choice',
            'galleries': 'Implicit feature; users may not know about automatic gallery rendering'
        }

        if feature_name in hypotheses:
            return hypotheses[feature_name]

        if usage_pct < 2:
            return 'Very low usage; feature may be unknown or have limited use cases'
        elif usage_pct < 5:
            return 'Low usage; may need better documentation or tutorials'
        else:
            return 'Moderate-low usage; consider improving discoverability'

    def _save_report(self, results):
        """Save human-readable report."""
        report_file = self.cluster_output_name.with_suffix('.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LiaScript Document Cluster Analysis\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Documents Analyzed: {results['total_documents']}\n")
            f.write(f"Analysis Timestamp: {results['timestamp']}\n\n")

            # Cluster Statistics
            f.write("-" * 80 + "\n")
            f.write("Document Clusters (documents may belong to multiple clusters)\n")
            f.write("-" * 80 + "\n")

            for cluster, data in sorted(results['cluster_statistics'].items(),
                                        key=lambda x: x[1]['count'], reverse=True):
                f.write(f"{cluster:25s}: {data['count']:5d} documents ({data['percentage']:6.2f}%)\n")

            # Cluster Profiles
            f.write("\n" + "-" * 80 + "\n")
            f.write("Cluster Profiles (Top 5 Features per Cluster)\n")
            f.write("-" * 80 + "\n")

            for cluster, profile in results['cluster_profiles'].items():
                f.write(f"\n{cluster.upper()} ({profile['document_count']} docs):\n")
                f.write("  Top Features:\n")
                for feature, pct in list(profile['top_features'].items())[:5]:
                    f.write(f"    - {feature}: {pct}%\n")

                # Show official templates for this cluster
                if 'official_templates' in profile and profile['official_templates']:
                    f.write("  Official LiaTemplates:\n")
                    for template, data in profile['official_templates'].items():
                        f.write(f"    - {template}: {data['count']} docs ({data['percentage']}%)\n")

                # Show template categories for this cluster
                if 'template_categories' in profile and profile['template_categories']:
                    f.write("  Template Categories:\n")
                    for category, data in list(profile['template_categories'].items())[:3]:
                        f.write(f"    - {category}: {data['count']} docs ({data['percentage']}%)\n")

            # Official LiaTemplates Summary
            if 'official_templates' in results and results['official_templates']['template_usage']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("Official LiaTemplates Usage (from github.com/LiaTemplates)\n")
                f.write("-" * 80 + "\n")

                f.write("\nTop Templates:\n")
                for template, data in list(results['official_templates']['template_usage'].items())[:15]:
                    f.write(f"  {template:25s}: {data['count']:5d} docs ({data['percentage']:6.2f}%) - {data['category']}\n")

                f.write("\nTemplate Categories:\n")
                for category, data in results['official_templates']['category_usage'].items():
                    f.write(f"  {category:25s}: {data['count']:5d} docs ({data['percentage']:6.2f}%)\n")

            # Complexity Metrics
            f.write("\n" + "-" * 80 + "\n")
            f.write("Complexity Metrics\n")
            f.write("-" * 80 + "\n")

            complexity = results['complexity_metrics']
            f.write(f"Average features per document: {complexity['avg_features_per_doc']}\n")
            f.write(f"Median features per document: {complexity['median_features_per_doc']}\n")
            f.write(f"Max features in a document: {complexity['max_features_per_doc']}\n")

            # Documentation Gaps
            f.write("\n" + "-" * 80 + "\n")
            f.write("Potential Documentation Gaps (Features < 10% usage)\n")
            f.write("-" * 80 + "\n")

            for feature, data in results['documentation_gaps'].items():
                f.write(f"\n{feature} ({data['usage_pct']}%):\n")
                f.write(f"  Hypothesis: {data['hypothesis']}\n")

            # Author-based Analysis
            if 'author_analysis' in results:
                author = results['author_analysis']
                f.write("\n" + "=" * 80 + "\n")
                f.write("AUTHOR-BASED ANALYSIS (avoiding single-author bias)\n")
                f.write("=" * 80 + "\n")

                f.write(f"\nTotal Authors: {author['total_authors']}\n")
                f.write(f"Total Documents: {results['total_documents']}\n")
                f.write(f"Avg Documents per Author: {results['total_documents'] / author['total_authors']:.1f}\n\n")

                # Author distribution
                f.write("-" * 80 + "\n")
                f.write("Author Distribution\n")
                f.write("-" * 80 + "\n")
                dist = author['author_distribution']
                f.write(f"  Authors with 1 document:      {dist.get('authors_with_1_doc', 0):5d} ({dist.get('authors_with_1_doc', 0) / author['total_authors'] * 100:.1f}%)\n")
                f.write(f"  Authors with 2-10 documents:  {dist.get('authors_with_2_10_docs', 0):5d} ({dist.get('authors_with_2_10_docs', 0) / author['total_authors'] * 100:.1f}%)\n")
                f.write(f"  Authors with 11-50 documents: {dist.get('authors_with_11_50_docs', 0):5d} ({dist.get('authors_with_11_50_docs', 0) / author['total_authors'] * 100:.1f}%)\n")
                f.write(f"  Authors with 51-100 documents:{dist.get('authors_with_51_100_docs', 0):5d} ({dist.get('authors_with_51_100_docs', 0) / author['total_authors'] * 100:.1f}%)\n")
                f.write(f"  Authors with 100+ documents:  {dist.get('authors_with_100plus_docs', 0):5d} ({dist.get('authors_with_100plus_docs', 0) / author['total_authors'] * 100:.1f}%)\n")

                # Template adoption by authors
                f.write("\n" + "-" * 80 + "\n")
                f.write("Official Template Adoption (by unique authors)\n")
                f.write("-" * 80 + "\n")
                f.write("\nComparison: Document-based vs Author-based metrics\n\n")
                f.write(f"{'Template':<25} {'Docs':>8} {'Doc%':>8} {'Authors':>8} {'Author%':>8}\n")
                f.write("-" * 60 + "\n")

                # Compare document-based vs author-based
                doc_templates = results['official_templates']['template_usage']
                author_templates = author['template_adoption']

                for template in list(doc_templates.keys())[:15]:
                    doc_data = doc_templates[template]
                    auth_data = author_templates.get(template, {'author_count': 0, 'adoption_pct': 0})
                    f.write(f"{template:<25} {doc_data['count']:>8} {doc_data['percentage']:>7.1f}% {auth_data['author_count']:>8} {auth_data['adoption_pct']:>7.1f}%\n")

                # Category adoption by authors
                f.write("\n" + "-" * 80 + "\n")
                f.write("Template Category Adoption (by unique authors)\n")
                f.write("-" * 80 + "\n")
                for category, data in author['category_adoption'].items():
                    f.write(f"  {category:25s}: {data['author_count']:5d} authors ({data['adoption_pct']:6.2f}%)\n")

                # Feature adoption by authors
                f.write("\n" + "-" * 80 + "\n")
                f.write("Feature Adoption (by unique authors, top 20)\n")
                f.write("-" * 80 + "\n")
                for feature, data in list(author['feature_adoption'].items())[:20]:
                    f.write(f"  {feature:25s}: {data['author_count']:5d} authors ({data['adoption_pct']:6.2f}%)\n")

                # Top author profiles
                f.write("\n" + "-" * 80 + "\n")
                f.write("Top 10 Author Profiles\n")
                f.write("-" * 80 + "\n")
                for author_name, profile in author['author_profiles'].items():
                    f.write(f"\n{author_name} ({profile['document_count']} docs):\n")
                    f.write(f"  Templates used ({profile['template_count']}): {', '.join(profile['templates_used'][:5])}\n")
                    if len(profile['templates_used']) > 5:
                        f.write(f"    ... and {len(profile['templates_used']) - 5} more\n")
                    f.write(f"  Categories: {', '.join(profile['categories_used'])}\n")

            # Author Clusters (K-Means with features AND templates)
            if 'author_clusters' in author and author['author_clusters']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("AUTHOR CLUSTERS (K-Means with Features + Templates)\n")
                f.write("=" * 80 + "\n")

                clusters_info = author['author_clusters']
                f.write(f"\nClustering Method: K-Means with {clusters_info.get('n_features_used', 'N/A')} features\n")
                f.write(f"Features used: {clusters_info.get('feature_types', 'N/A')}\n")
                f.write(f"Authors clustered: {clusters_info.get('authors_clustered', 'N/A')} (min 2 documents)\n\n")

                for cluster_id, cluster_data in clusters_info.get('clusters', {}).items():
                    f.write(f"\n--- {cluster_data.get('name', f'Cluster {cluster_id}')} ---\n")
                    f.write(f"  Authors: {cluster_data['author_count']}\n")
                    f.write(f"  Total Documents: {cluster_data['total_docs']}\n")

                    f.write("  Top Features:\n")
                    for feat, pct in list(cluster_data.get('top_features', {}).items())[:5]:
                        f.write(f"    - {feat}: {pct:.1f}%\n")

                    if cluster_data.get('top_templates'):
                        f.write("  Top Templates:\n")
                        for tmpl, pct in list(cluster_data['top_templates'].items())[:5]:
                            f.write(f"    - {tmpl}: {pct:.1f}%\n")

                    f.write(f"  Representative Authors: {', '.join(cluster_data.get('representative_authors', [])[:5])}\n")

        logging.info(f"Report saved to {report_file}")
