# this document makes professional-quality graphs with Altair
import pandas as pd
import altair as alt

# demo data
# In production, this would be: df = pd.read_csv("data/processed_results.csv")
df = pd.DataFrame({
    "epoch": [1, 2, 3, 4, 5] * 2,
    "loss": [0.9, 0.6, 0.4, 0.2, 0.1, 1.2, 0.8, 0.5, 0.3, 0.2],
    "model_variant": ["ResNet", "ResNet", "ResNet", "ResNet", "ResNet", 
                      "Transformer", "Transformer", "Transformer", "Transformer", "Transformer"]
})

def theme(color_scheme='tableau10'):
    """ Creates an arial theme for Vega-Altair graphs based on an input color scheme
    Args:
        color_scheme: (string) name of a Vega-Altair color scheme (default is 'tableau10')
    Returns:
        Thematic dictionary
    """
    return {
        'config': {
            'view': {'stroke': 'transparent'}, # Removes the box around the chart
            'title': {'fontSize': 16, 'font': 'Arial', 'anchor': 'start', 'color': '#111111'},
            'axis': {
                'labelFont': 'Arial',
                'titleFont': 'Arial',
                'gridColor': '#EFEFEF',
                'tickColor': '#CCCCCC'
            },
            'range': {
                'category': {'scheme': color_scheme}
            }
        }
    }

def line_chart(df, savepath, scheme='tableau10'):
    """ Makes a professional line chart
    Args:
        df: a Pandas dataframe containing
        savepath: save destination
        theme: the color scheme and text style of the graph ('tableau10', 'category20c', etc...)
    
    Outputs:
        Saves the chart to the savepath
    """
    # register and enable the theme
    alt.themes.register('theme', lambda: theme(scheme))
    alt.themes.enable('theme')

    # prepare the graph
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('epoch:Q', title='Training Epoch'),
        y=alt.Y('loss:Q', title='Validation Loss'),
        color=alt.Color('model_variant:N', title='Architecture'), # :N means Nominal (categories)
    ).properties(
        width=600,
        height=400,
        title="Model Convergence Comparison"
    )
    chart.save(savepath+'/model_loss_comparison.pdf')

line_chart(df, 'graphs')