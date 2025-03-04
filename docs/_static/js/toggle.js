document.addEventListener('DOMContentLoaded', function() {
    const toggleButton = document.createElement('button');
    toggleButton.id = 'toggle-docs';
    toggleButton.textContent = 'Switch to CLI Docs';
    toggleButton.style.position = 'fixed';
    toggleButton.style.bottom = '20px';
    toggleButton.style.right = '20px';
    toggleButton.style.zIndex = '1000';
    toggleButton.style.padding = '10px 20px';
    toggleButton.style.backgroundColor = '#007BFF';
    toggleButton.style.color = '#FFF';
    toggleButton.style.border = 'none';
    toggleButton.style.borderRadius = '5px';
    toggleButton.style.cursor = 'pointer';

    document.body.appendChild(toggleButton);

    toggleButton.addEventListener('click', function() {
        const pythonDocs = document.querySelectorAll('.python-docs');
        const cliDocs = document.querySelectorAll('.cli-docs');

        pythonDocs.forEach(doc => doc.style.display = doc.style.display === 'none' ? 'block' : 'none');
        cliDocs.forEach(doc => doc.style.display = doc.style.display === 'none' ? 'block' : 'none');

        toggleButton.textContent = toggleButton.textContent === 'Switch to CLI Docs' ? 'Switch to Python Docs' : 'Switch to CLI Docs';
    });
});