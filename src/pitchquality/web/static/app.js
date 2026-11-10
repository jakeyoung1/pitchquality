// The pages work without this file. It only saves a click on the filters.
document.querySelectorAll("select[data-autosubmit]").forEach(function (select) {
  select.addEventListener("change", function () {
    if (select.form) select.form.requestSubmit();
  });
});
