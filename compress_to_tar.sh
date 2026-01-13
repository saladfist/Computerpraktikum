mkdir -p ./team_7
tar --exclude='./cluster-data' --exclude='./__pycache__' --exclude='./cluster-results' --exclude='./notes.txt' --exclude='./requirements.txt' -czvf ./team_7_programs.tar.gz -C programs .
mv ./team_7_programs.tar.gz ./team_7/