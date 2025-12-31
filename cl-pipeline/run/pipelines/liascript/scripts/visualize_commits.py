"""
Interactive visualization of LiaScript commit statistics.

This script creates an interactive scatter plot showing:
- X-axis: Duration in days (logarithmic scale)
- Y-axis: Number of commits (logarithmic scale)
- Clickable points that open the document URL
- Hover information with authors and URL
"""

import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def load_commit_data(commits_file_path):
    """Load commit data from pickle file."""
    commits_file = Path(commits_file_path)

    if not commits_file.exists():
        raise FileNotFoundError(f"Commits file not found: {commits_file}")

    df = pd.read_pickle(commits_file)
    print(f"Loaded {len(df)} LiaScript documents")

    return df


def prepare_data(df):
    """Prepare data for visualization."""
    # Filter rows with valid date information
    df_valid = df.dropna(subset=['first_commit', 'last_commit']).copy()

    # Calculate duration in days
    df_valid['duration_days'] = (df_valid['last_commit'] - df_valid['first_commit']).dt.days

    # Filter out entries with 0 commits or 0 duration (add small offset for log scale)
    df_valid = df_valid[df_valid['commit_count'] > 0].copy()
    df_valid['duration_days_safe'] = df_valid['duration_days'].apply(lambda x: max(x, 0.1))

    # Extract repository name from URL for better labeling
    df_valid['repo_name'] = df_valid['file_download_url'].apply(
        lambda x: '/'.join(x.split('/')[3:5]) if 'github' in x else x
    )

    # Create author list as string
    df_valid['authors_str'] = df_valid['contributors_list'].apply(
        lambda x: ', '.join(list(dict.fromkeys(x))) if isinstance(x, list) and len(x) > 0 else 'Unknown'
    )

    # Truncate long author lists for display
    df_valid['authors_display'] = df_valid['authors_str'].apply(
        lambda x: x[:100] + '...' if len(x) > 100 else x
    )

    print(f"Prepared {len(df_valid)} documents for visualization")

    return df_valid


def aggregate_overlapping_points(df):
    """
    Aggregate documents that have the same commit count and duration.
    Returns a dataframe with aggregated data and grouping information.
    """
    # Group by commit_count and duration_days_safe
    grouped = df.groupby(['commit_count', 'duration_days_safe']).agg({
        'file_download_url': list,
        'repo_name': list,
        'author_count': 'mean',  # Average for color coding
        'authors_str': list,
        'first_commit': 'min',
        'last_commit': 'max',
        'duration_days': 'first'
    }).reset_index()

    # Add count of documents per group
    grouped['doc_count'] = grouped['file_download_url'].apply(len)

    # Calculate marker size based on doc_count
    # Single documents: size 8, groups scale up to size 30
    grouped['marker_size'] = grouped['doc_count'].apply(
        lambda x: 8 if x == 1 else min(8 + (x * 2), 30)
    )

    print(f"Aggregated {len(df)} documents into {len(grouped)} unique positions")
    print(f"Groups with multiple documents: {len(grouped[grouped['doc_count'] > 1])}")
    print(f"Largest group: {grouped['doc_count'].max()} documents")

    return grouped


def create_hover_text_single(row):
    """Create hover text for a single document."""
    return (
        f"<b>{row['repo_name'][0]}</b><br>"
        f"Commits: {row['commit_count']}<br>"
        f"Duration: {row['duration_days']} days ({row['duration_days']/365:.1f} years)<br>"
        f"Authors ({int(row['author_count'])}): {row['authors_str'][0][:100]}<br>"
        f"Period: {row['first_commit'].date()} to {row['last_commit'].date()}<br>"
        f"<br>URL: {row['file_download_url'][0]}<br>"
        f"<i>Click to open URL</i>"
    )


def create_hover_text_group(row):
    """Create hover text for a group of documents."""
    doc_list = "<br>".join([f"  • {repo}" for repo in row['repo_name'][:10]])  # Show first 10
    if len(row['repo_name']) > 10:
        doc_list += f"<br>  • ... and {len(row['repo_name']) - 10} more"

    return (
        f"<b>GROUP: {row['doc_count']} Documents</b><br>"
        f"Commits: {row['commit_count']}<br>"
        f"Duration: {row['duration_days']} days ({row['duration_days']/365:.1f} years)<br>"
        f"<br>Documents:<br>{doc_list}<br>"
        f"<i>Multiple documents at this position</i>"
    )


def create_interactive_plot(df):
    """Create interactive scatter plot with size-based aggregation."""

    # Aggregate overlapping points
    df_agg = aggregate_overlapping_points(df)

    # Create hover text based on whether it's a single document or group
    hover_text = []
    for idx, row in df_agg.iterrows():
        if row['doc_count'] == 1:
            hover_text.append(create_hover_text_single(row))
        else:
            hover_text.append(create_hover_text_group(row))

    # For single documents, store URL for clicking; for groups, use empty string
    click_urls = []
    for idx, row in df_agg.iterrows():
        if row['doc_count'] == 1:
            click_urls.append(row['file_download_url'][0])
        else:
            click_urls.append('')  # No single URL for groups

    # Create the scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_agg['duration_days_safe'],
        y=df_agg['commit_count'],
        mode='markers',
        marker=dict(
            size=df_agg['marker_size'],
            color=df_agg['author_count'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Avg. Number<br>of Authors"),
            line=dict(width=0.5, color='white'),
            opacity=0.7
        ),
        text=hover_text,
        hoverinfo='text',
        customdata=click_urls,
        name='LiaScript Documents'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Duration and Commit Count of LiaScript Materials<br><sub>Point size indicates number of documents. Click single documents to open URL.</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Duration (days)',
            type='log',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Number of Commits',
            type='log',
            gridcolor='lightgray'
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=1200,
        height=800,
        font=dict(size=12),
        clickmode='event+select'
    )

    return fig


def main():
    """Main function to create and display the visualization."""

    # Path to commits data
    commits_file = '/media/sz/Data/Connected_Lecturers/LiaScript/raw/LiaScript_commits.p'

    # Load and prepare data
    print("Loading commit data...")
    df = load_commit_data(commits_file)

    print("Preparing data for visualization...")
    df_prepared = prepare_data(df)

    # Create statistics summary
    print("\n=== Statistics ===")
    print(f"Total documents: {len(df_prepared)}")
    print(f"Total commits: {df_prepared['commit_count'].sum()}")
    print(f"Average commits per document: {df_prepared['commit_count'].mean():.1f}")
    print(f"Median duration: {df_prepared['duration_days'].median():.0f} days")
    print(f"Max duration: {df_prepared['duration_days'].max():.0f} days ({df_prepared['duration_days'].max()/365:.1f} years)")
    print(f"Max commits: {df_prepared['commit_count'].max()}")

    # Create interactive plot
    print("\nCreating interactive visualization...")
    fig = create_interactive_plot(df_prepared)

    # Save to HTML file with JavaScript for clickable URLs
    output_file = Path(__file__).parent / 'liascript_commits_visualization.html'

    # Create standalone HTML that can be sent via email
    # include_plotlyjs='cdn' uses CDN (smaller file, needs internet)
    # include_plotlyjs=True embeds full plotly (larger file, works offline)
    html_string = fig.to_html(
        include_plotlyjs='cdn',  # Use CDN for smaller file size
        full_html=True,
        config={'displayModeBar': True, 'responsive': True}
    )

    # Add JavaScript to make points clickable - insert before closing body tag
    click_script = """
<script>
// Wait for Plotly to be fully loaded
window.addEventListener('load', function() {
    // Give Plotly a moment to initialize
    setTimeout(function() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {
            console.log('Plotly div found, adding click handler');
            plotDiv.on('plotly_click', function(data) {
                console.log('Click detected', data);
                var point = data.points[0];
                if (point && point.customdata) {
                    var url = point.customdata;
                    if (url && url !== '') {
                        console.log('Opening URL:', url);
                        window.open(url, '_blank');
                    } else {
                        console.log('This is a group - no URL to open');
                    }
                }
            });
        } else {
            console.error('Plotly div not found');
        }
    }, 1000);
});
</script>
"""

    # Insert JavaScript before </body>
    html_string = html_string.replace('</body>', click_script + '\n</body>')

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_string)

    print(f"\nVisualization saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\nThis HTML file can be:")
    print("  - Opened directly in any web browser")
    print("  - Sent via email (requires internet connection to load Plotly)")
    print("  - Shared via cloud storage or file sharing")

    # Also create an offline version with embedded Plotly (larger but works without internet)
    output_file_offline = Path(__file__).parent / 'liascript_commits_visualization_offline.html'
    html_string_offline = fig.to_html(
        include_plotlyjs=True,  # Embed full Plotly library
        full_html=True,
        config={'displayModeBar': True, 'responsive': True}
    )
    html_string_offline = html_string_offline.replace('</body>', click_script + '\n</body>')

    with open(output_file_offline, 'w', encoding='utf-8') as f:
        f.write(html_string_offline)

    print(f"\nOffline version saved to: {output_file_offline}")
    print(f"File size: {output_file_offline.stat().st_size / (1024*1024):.1f} MB")
    print("This version works without internet but is larger.")

    # Show in browser
    print("\nOpening visualization in browser...")
    fig.show()


if __name__ == '__main__':
    main()
