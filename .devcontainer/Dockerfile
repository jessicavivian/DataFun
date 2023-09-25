FROM python:3.8

# Set the working directory
WORKDIR /workspace

# Install Jupyter Notebook, Pandas, Matplotlib, TextBlob, and WordCloud
RUN pip install jupyter pandas matplotlib textblob wordcloud

#Expose the Jupyter Notebook port
EXPOSE 8888

#Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]